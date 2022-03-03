
import torch


from datasets import *
from models.networks import MaterialEditGeneratorWithNormals2Steps,MaterialEditGeneratorWithNormals

from utils.im_util import denorm, write_labels_on_images
import numpy as np
from agents.MaterialEditNet import MaterialEditNet
from utils import resize_right, interp_methods



class MaterialEditNetWithNormals2Steps(MaterialEditNet):
    def __init__(self, config):
        super(MaterialEditNetWithNormals2Steps, self).__init__(config)

        ###only change generator
        self.G = MaterialEditGeneratorWithNormals2Steps(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv, n_concat_normals=config.n_concat_normals, normalization=self.norm, first_conv=config.first_conv, n_bottlenecks=config.n_bottlenecks, img_size=self.config.image_size, batch_size=self.config.batch_size)
        print(self.G)

        #load step1 network
        self.G_small = MaterialEditGeneratorWithNormals(conv_dim=32,n_layers=6,max_dim=512, im_channels=config.img_channels,  attr_dim=len(config.attrs), n_attr_deconv=1, n_concat_normals=4, normalization=self.norm, n_bottlenecks=2, first_conv=False)
        #print(self.G_small) 
        self.load_model_from_path(self.G_small,config.g1_checkpoint)
        self.G_small.eval()

        self.logger.info("MaterialEditNet with normals in 2 steps ready")




    ################################################################
    ##################### EVAL UTILITIES ###########################
    def decode(self,bneck,att):
        normals= self.get_normals()
        _, fn_features = self.get_step1_output(att)
        return self.G.decode(att,bneck,normals,fn_features)


    def init_sample_grid(self):
        # first line to also put the output from net1 in the grid
        x_fake_list = [self.get_step1_output(self.batch_a_att)[0],self.batch_Ia[:,:3]]
        return x_fake_list


    def get_normals(self):
        return self.batch_normals[:,:3]


    def get_step1_output(self,att):
        with torch.no_grad(): #G1 is not optimized, no need for gradients
            if self.config.image_size == 128:
                _,z = self.G_small.encode(self.batch_Ia)
                fn_output, fn_features= self.G_small.decode_with_features(att,z,self.get_normals())
                return fn_output, fn_features
            else: # self.config.image_size == 256:
                #rescale to 128 before going through G1
                rescaled_im=resize_right.resize(self.batch_Ia, scale_factors=0.5)
                rescaled_normals=resize_right.resize(self.get_normals(), scale_factors=0.5)

                _,z = self.G_small.encode(rescaled_im)
                fn_output, fn_features = self.G_small.decode_with_features(att,z,rescaled_normals)
            
                fn_output=resize_right.resize(fn_output, scale_factors=2)
                return fn_output, fn_features



