import os
import logging
import time
import datetime
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.misc import print_cuda_statistics
import numpy as np

###### things to be shared in a TrainingModule
class TrainingModule(object):
    def __init__(self, config):
        #setup CUDA, seeds, logger...
        self.config = config
        self.logger = logging.getLogger(config.exp_name)
        self.logger.info("Creating architecture...")

        # deterministic behaviour
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)

        self.data_loader = None #TODO define in children

        self.current_iteration = 0
        self.cuda = torch.cuda.is_available() & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.writer = SummaryWriter(log_dir=self.config.summary_dir)


    ################################################################
    ###################### SAVE/lOAD ###############################
    def save_one_model(self,model,optimizer,name):
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir, '{}_{}.pth.tar'.format(name,self.current_iteration)))

    def load_one_model(self,model,optimizer,name, iter=None):
        if iter == None: iter=self.config.checkpoint
        checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, '{}_{}.pth.tar'.format(name,iter)),map_location=self.device)
        to_load = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(to_load)
        model.to(self.device)
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer'])
    def load_model_from_path(self,model,path):
        checkpoint = torch.load(path,map_location=self.device)
        to_load = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(to_load)
        model.to(self.device)

    ################################################################
    ################### OPTIM UTILITIES ############################
    def build_optimizer(self,model,lr):
        model=model.to(self.device)
        return optim.Adam(model.parameters(), lr, [self.config.beta1, self.config.beta2])
    def build_scheduler(self,optimizer,not_load=False):
        last_epoch=-1 if (self.config.checkpoint == None or not_load) else self.config.checkpoint
        return optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_iters, gamma=0.5, last_epoch=last_epoch)
    def optimize(self,optimizer,loss): 
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


    ################################################################
    ################### LOSSES UTILITIES ###########################
    def regression_loss(self, logit, target): #static
        return F.l1_loss(logit,target)/ logit.size(0)
    def classification_loss(self, logit, target): #static
        return F.cross_entropy(logit,target) 
    def image_reconstruction_loss(self, Ia, Ia_hat):
        if self.config.rec_loss == 'l1':
            g_loss_rec = F.l1_loss(Ia,Ia_hat)
        elif self.config.rec_loss == 'l2':
            g_loss_rec = F.mse_loss(Ia,Ia_hat)
        elif self.config.rec_loss == 'perceptual':
            ### WARNING: need to be created
            l1_loss=F.l1_loss(Ia,Ia_hat)
            self.scalars['G/loss_rec_l1'] = l1_loss.item()
            g_loss_rec = l1_loss
            #add perceptual loss
            f_img = self.vgg16_f(Ia)
            f_img_hat = self.vgg16_f(Ia_hat)
            if self.config.lambda_G_perc > 0:
                self.scalars['G/loss_rec_perc'] = self.config.lambda_G_perc * self.loss_P(f_img_hat, f_img)
                g_loss_rec += self.scalars['G/loss_rec_perc'].item()
            if self.config.lambda_G_style > 0:
                self.scalars['G/loss_rec_style'] = self.config.lambda_G_style * self.loss_S(f_img_hat, f_img)
                g_loss_rec += self.scalars['G/loss_rec_style'].item()
        return g_loss_rec
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    ################################################################
    ##################### MAIN FUNCTIONS ###########################
    def run(self):
        assert self.config.mode in ['train', 'test']
        try:
            if self.config.mode == 'train':
                self.train()
            else:
                self.test()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        except Exception as e:
            log_file = open(os.path.join(self.config.log_dir, 'exp_error.log'), 'w+')
            traceback.print_exc(file=log_file)
            traceback.print_exc()
        finally:
            self.finalize()

    def batch_to_device(self,batch):
        return [elem.to(self.device) if isinstance(elem, torch.Tensor) else elem for elem in batch]

    def train(self):
        self.setup_all_optimizers()
        self.writer.add_hparams(dict(self.config),{"dumb_val":0})

        # samples used for testing the net
        val_iter = iter(self.data_loader.val_loader)
        val_data = self.batch_to_device(next(val_iter))


        data_iter = iter(self.data_loader.train_loader)
        start_time = time.time()
        start_batch = self.current_iteration // self.data_loader.train_iterations
        tdqm_post={"Time":"0"}
        for batch in range(start_batch, self.config.max_epoch):
            tdqm_post["Time"]=str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
            tqdm_loader = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations, desc='Batch {} (iter {})'.format(batch,self.current_iteration),postfix=tdqm_post,leave=(batch==self.config.max_epoch-1))

            for train_data in tqdm_loader: #tdqm
                #train
                self.training_mode()
                ##################### TRAINING STEP
                self.scalars = {}
                self.training_step(self.batch_to_device(train_data))
                ###################################
                self.current_iteration += 1
                self.step_schedulers()

                # print summary on terminal and on tensorboard
                if self.current_iteration % self.config.summary_step == 0:
                    tdqm_post["Time"]=str(datetime.timedelta(seconds=time.time() - start_time))[:-7]
                    tqdm_loader.set_postfix(tdqm_post)
                    for tag, value in self.scalars.items():
                        self.writer.add_scalar(tag, value, self.current_iteration)

                # sample
                if (self.current_iteration) % self.config.sample_step == 0 or self.current_iteration==1:
                    self.eval_mode()
                    with torch.no_grad():
                        self.validating_step(val_data)

                # save checkpoint
                if self.current_iteration % self.config.checkpoint_step == 0:
                    self.save_checkpoint()
        self.save_checkpoint()

    def test(self):
        self.load_checkpoint()

        tqdm_loader = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc='Testing at checkpoint {}'.format(self.config.checkpoint))

        self.eval_mode()
        with torch.no_grad():
            for i, test_data in enumerate(tqdm_loader):
                self.testing_step(self.batch_to_device(test_data),i)


    def finalize(self):
        print('Please wait while finalizing the operation.. Thank you')
        if self.config.mode == "train": 
            self.save_checkpoint()
        self.writer.export_scalars_to_json(os.path.join(self.config.summary_dir, 'all_scalars.json'))
        self.writer.close()


    ####################################################################################
    #####################        TBD IN CHILD CLASSES        ###########################
    #load/save/init all models and optimizers
    def save_checkpoint(self):
        raise NotImplementedError
    def load_checkpoint(self):
        raise NotImplementedError
    def setup_all_optimizers(self):
        raise NotImplementedError
    #step scheduler and log them into dict
    def step_schedulers(self):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError
    def validating_step(self, batch):
        raise NotImplementedError
    def testing_step(self, batch, batch_id):
        raise NotImplementedError

    def eval_mode(self):
        raise NotImplementedError
    def training_mode(self):
        raise NotImplementedError



