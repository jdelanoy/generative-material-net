

from models.networks import MaterialEditGeneratorWithNormals

import numpy as np
from agents.MaterialEditNet import MaterialEditNet



class MaterialEditNetWithNormals(MaterialEditNet):
    def __init__(self, config):
        super(MaterialEditNetWithNormals, self).__init__(config)

        ###only change generator
        self.G = MaterialEditGeneratorWithNormals(conv_dim=config.g_conv_dim,n_layers=config.g_layers,max_dim=config.max_conv_dim, im_channels=config.img_channels, attr_dim=len(config.attrs), n_attr_deconv=config.n_attr_deconv, n_concat_normals=config.n_concat_normals, normalization=self.norm, first_conv=config.first_conv, n_bottlenecks=config.n_bottlenecks, img_size=self.config.image_size, batch_size=self.config.batch_size)
        print(self.G)

        self.logger.info("MaterialEditNet with normals ready")




    ################################################################
    ##################### EVAL UTILITIES ###########################
    def decode(self,bneck,att):
        normals= self.get_normals()
        return self.G.decode(att,bneck,normals)

    def init_sample_grid(self):
        x_fake_list = [self.get_normals(),self.batch_Ia[:,:3]]
        return x_fake_list

    def get_normals(self):
        return self.batch_normals[:,:3]









