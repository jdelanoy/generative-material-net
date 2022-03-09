import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

import pix2normal.resnet_model as resnet_model
import numpy as np

import torchvision.models as models
from collections import OrderedDict
import functools



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# References: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py
#             https://codecov.io/gh/PyTorchLightning/lightning-bolts/src/511325d8e35385ee7370f87b4e1459578a9616f0/pl_bolts/models/gans/pix2pix/pix2pix_module.py
#             https://github.com/znxlwm/pytorch-pix2pix/blob/master/pytorch_pix2pix.py
#             https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb
###############################################################################
class PerceptualLoss(nn.Module):
    def __init__(self, loss_fn=F.l1_loss):
        super(PerceptualLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, x_act, y_act):
        # x_act and y_act are dicts on the following form {layer_name: features}
        loss = 0
        for k in x_act.keys():
            loss += self.loss_fn(x_act[k], y_act[k].detach())
        return loss / len(x_act)

class VGG16FeatureExtractor(nn.Module):
    __layer_names = [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'maxpool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'maxpool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'maxpool5'
    ]

    def __init__(self, layers, use_avg_pool=False, rescale_img=False,
                 random_crop_image=False, normalize=False):

        super(VGG16FeatureExtractor, self).__init__()

        # remove unused layers from vgg16
        vgg16_extractor = OrderedDict()
        vgg16 = models.vgg16(pretrained=True).eval().features
        l_partial = layers.copy()
        for l_name, l in zip(self.__layer_names, vgg16):
            # store only necessary layers
            if len(l_partial) == 0:
                break

            vgg16_extractor[l_name] = l
            if l_name in l_partial:
                l.register_forward_hook(functools.partial(self._save, l_name))
                l_partial.remove(l_name)

        # set avg pooling if required
        if use_avg_pool:
            for name, mod in vgg16_extractor.named_modules():
                if 'pool' in name:
                    setattr(vgg16_extractor, name, nn.AvgPool2d(2, 2))

        # set normalization and reescaling values
        self.rescale = rescale_img
        self.normalize = normalize
        self.norm = ImageNetInputNorm()
        self.random_crop = random_crop_image
        self.crop_coord = []

        # create model and dict to store the activations
        self.vgg16_extractor = nn.Sequential(vgg16_extractor)
        self.activations = {}
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.rescale:
            x = F.interpolate(x, size=(224, 224), mode='nearest')
        if self.random_crop:
            if len(self.crop_coord) == 0:
                self.crop_coord = random_crop_ixs(x, (224, 224))
            x = trf_F.crop(x, *self.crop_coord)
        if self.normalize:
            x = self.norm(x)

        self.vgg16_extractor(x)
        activations = self.activations
        self.activations = {}  # clean up the variable
        return activations

    def _save(self, name, module, input, output):
        self.activations[name] = output

    def reset_crop_coord(self):
        self.crop_coord = []

class ImageNetInputNorm(nn.Module):
    """
    Normalize images channels as torchvision models expects, in a
    differentiable way
    """

    def __init__(self):
        super(ImageNetInputNorm, self).__init__()
        self.register_buffer(
            'norm_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))

        self.register_buffer(
            'norm_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input):
        return (input - self.norm_mean) / self.norm_std

class pix2normal(pl.LightningModule):
    def __init__(
        self,
        num_filters,
        input_channels,
        input_height,
        input_width,
        learning_rate=0.01,
        lambda_scale_loss=10,
        gan_option=True,
        perceptual_loss=True

    ):
        super().__init__()

        self.save_hyperparameters()
        self.lambda_scale = lambda_scale_loss
        self.perceptual_loss_opt = perceptual_loss
        self.lr = learning_rate
        self.gan_option = gan_option
        self.img_dim = (input_channels, input_height, input_width)

        self.fext = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        self.percept_loss = PerceptualLoss()#nn.MSELoss()
        self.generator = self.init_generator(num_filters)
        self.discriminator = self.init_discriminator(num_filters)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        #self.recon_criterion = nn.L1Loss()
        self.recon_criterion = nn.MSELoss() #maybe l2 (MSE)?

        #self.data_loader = datasets.MaterialDataLoader()

    def init_generator(self, num_filters):
        generator = resnet_model.generator(d=num_filters)
        generator.weight_init(mean=0.0, std=0.02)
        return generator

    def init_discriminator(self, num_filters):
        discriminator = resnet_model.discriminator(d=num_filters)
        discriminator.weight_init(mean=0.0, std=0.02)
        return discriminator

    #revisar loss functions y steps

    def generator_step(self,real_normals, renders):
        gan_option = self.gan_option
        if (gan_option):
            fake_normals = self.generator(renders)
            real_normals = real_normals[:,:3,:,:]
            for i in range(len(fake_normals)):
                fake_normals[i] = fake_normals[i] * renders[i,3:,:,:]
                real_normals[i] = real_normals[i] * renders[i,3:,:,:]
            disc_logits = self.discriminator(renders,fake_normals)

            adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))
            reconstruction_loss = self.recon_criterion(fake_normals,real_normals)

            self.log("Adversarial Loss (BCE)", adversarial_loss)
            self.log("Reconstruction Loss (MSE)", reconstruction_loss)

            lambda_scale_loss = self.lambda_scale

            if(self.perceptual_loss_opt): 
                f_fake = self.fext(fake_normals)
                f_real = self.fext(real_normals)

                perceptual_loss = self.percept_loss(f_fake,f_real)

                self.log("Perceptual Loss", perceptual_loss)
        
                return 0.25*adversarial_loss + lambda_scale_loss*reconstruction_loss + 0.1*lambda_scale_loss*perceptual_loss
            else:
                return adversarial_loss + lambda_scale_loss*reconstruction_loss
        else: 
            fake_normals = self.generator(renders)
            real_normals = real_normals[:,:3,:,:]
            for i in range(len(fake_normals)):
                fake_normals[i] = fake_normals[i] * renders[i,3:,:,:]
                real_normals[i] = real_normals[i] * renders[i,3:,:,:]
            reconstruction_loss = self.recon_criterion(fake_normals,real_normals)
            self.log("Reconstruction Loss (MSE)", reconstruction_loss)
        
            return reconstruction_loss 

        
    def discriminator_step(self,real_normals,renders):
        gan_option = self.gan_option
        if (gan_option):
            fake_normals = self.generator(renders)
            real_normals = real_normals[:,:3,:,:]
            for i in range(len(fake_normals)):
                fake_normals[i] = fake_normals[i] * renders[i,3:,:,:]
                real_normals[i] = real_normals[i] * renders[i,3:,:,:]
            fake_logits = self.discriminator(renders, fake_normals) 
            real_logits = self.discriminator(renders, real_normals) 

            fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
            real_loss = self.adversarial_criterion(real_logits, torch.zeros_like(real_logits))

            self.log("Disc Fake Loss (BCE)", fake_loss)
            self.log("Disc Real Loss (BCE)", real_loss)

            return (real_loss + fake_loss)/2.0
        else: 
            return 0

    def configure_optimizers(self):
        gan_option = self.gan_option
        lr = self.lr
        generator_optimizer = torch.optim.Adam(self.generator.parameters(),lr=lr)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=lr)
        lr_scheduler_gen = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(generator_optimizer),
            'monitor': 'Generator Train Loss',
            'reduce_on_plateau': True,
            'interval': 'epoch',
            'frequency': 1,
            'name': 'Generator Learning Rate',
            }
        lr_scheduler_disc = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer),
            'monitor': 'Discriminator Train Loss',
            'reduce_on_plateau': True,
            'interval': 'epoch',
            'frequency': 1,
            'name': 'Discriminator Learning Rate',
            }
        if (gan_option):
            return [discriminator_optimizer, generator_optimizer], [lr_scheduler_gen, lr_scheduler_disc]
        else: 
            return [generator_optimizer],[lr_scheduler_gen]

    def training_step(self, batch, batch_idx, optimizer_idx): #gan loss
        render, real_normals = batch

        if optimizer_idx == 0:
            loss = self.discriminator_step(real_normals,render)
            self.log('Discriminator Train Loss', loss)
        elif optimizer_idx == 1:
            loss = self.generator_step(real_normals, render)
            self.log('Generator Train Loss', loss)

        return loss

    #def training_step(self, batch, batch_idx): #no gan loss
    #    render, real_normals = batch
    #    loss = self.generator_step(real_normals, render)
    #    self.log('Generator Train Loss', loss)

    #    return loss


    def shared_step(self, batch, stage: str = 'val'):
        grid = []
        render, real_normals = batch
        fake_normals = self.generator(render)
        real_normals = real_normals[:,:3,:,:]
        for i in range(len(fake_normals)):
            fake_normals[i] = fake_normals[i] * render[i,3:,:,:]
            real_normals[i] = real_normals[i] * render[i,3:,:,:]
            render = render[:,:3,:,:]
        gan_option = self.gan_option

        if (gan_option):
            lambda_scale_loss = self.lambda_scale

            fake_logits = self.discriminator(render, fake_normals) 
            real_logits = self.discriminator(render, real_normals) 

            #losses
            adversarial_loss = self.adversarial_criterion(fake_logits, torch.ones_like(fake_logits))
            reconstruction_loss = self.recon_criterion(fake_normals,real_normals)
            if(self.perceptual_loss_opt):
                f_fake = self.fext(fake_normals)
                f_real = self.fext(real_normals)

                perceptual_loss = self.percept_loss(f_fake,f_real)

                generator_loss = 0.25*adversarial_loss + lambda_scale_loss*reconstruction_loss + 0.1*lambda_scale_loss*perceptual_loss
            else:
                generator_loss = adversarial_loss + lambda_scale_loss*reconstruction_loss

            fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
            real_loss = self.adversarial_criterion(real_logits, torch.zeros_like(real_logits))

            discriminator_loss = (fake_loss + real_loss)/2.0

            #log losses
            if (stage=='val'):
                self.log('Discriminator Val Loss', discriminator_loss)
                self.log('Generator Val Loss', generator_loss)
            else:
                self.log('Discriminator Test Loss', discriminator_loss)
                self.log('Generator Test Loss', generator_loss)
        else: 
            reconstruction_loss = self.recon_criterion(fake_normals,real_normals)

            generator_loss = reconstruction_loss

            #log losses
            if (stage=='val'):
                self.log('Generator Val Loss', generator_loss)
            else:
                self.log('Generator Test Loss', generator_loss)

        for i in range(12):
            rand_int = np.random.randint(0, len(render))
            tensor = torch.stack([render[rand_int], fake_normals[rand_int], real_normals[rand_int]])
            grid.append((tensor + 1) / 2)
            


    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')


    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')


    def forward(self,x):
        x = self.generator(x)
        return x
