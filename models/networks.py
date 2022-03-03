import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from utils import resize_right, interp_methods
from models.blocks import * 

    



# conv layers of the discriminator
def build_disc_layers(conv_dim=64, n_layers=6, max_dim = 512, in_channels = 3, activation='relu', normalization='batch',dropout=0):
    bias = normalization != 'batch'  # use bias only if we do not use a normalization layer  
    
    layers = []
    out_channels = conv_dim
    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        enc_layer=[ConvReluBn(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1,bias=bias,padding_mode='reflect'),activation,normalization=normalization if i > 0 else 'none')] 
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels=min(2*out_channels,max_dim)
    return layers


# conv layers of the encoder
def build_encoder_layers(conv_dim=64, n_layers=6, max_dim = 512, im_channels = 3, activation='relu', normalization='batch',dropout=0, first_conv=False):
    bias = normalization != 'batch'  # use bias only if we do not use a normalization layer 
    
    layers = []
    in_channels = im_channels
    out_channels = conv_dim
    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        kernel_size = 7 if (first_conv and i == 0) else 4
        enc_layer=[ConvReluBn(nn.Conv2d(in_channels, out_channels, kernel_size, stride=2 if (i>0 or not first_conv) else 1, padding=(kernel_size-1)//2, bias=bias,padding_mode='reflect'),activation,normalization=normalization)] 
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels=min(2*out_channels,max_dim)
    return layers


# encoder : conv layers + bottleneck
class Encoder(nn.Module):
    def __init__(self, conv_dim, n_layers, max_dim, im_channels, activation='relu', normalization='batch',first_conv=False, n_bottlenecks=2):
        super(Encoder, self).__init__()
        bias = normalization != 'batch'  # use bias only if we do not use a normalization layer 
        enc_layers=build_encoder_layers(conv_dim,n_layers,max_dim, im_channels,normalization=normalization,activation=activation, first_conv=first_conv) 
        self.encoder = nn.ModuleList(enc_layers)
        # residual blocks of bottlenect
        b_dim=min(max_dim,conv_dim * 2 ** (n_layers-1))
        self.bottleneck = nn.ModuleList([ResidualBlock(b_dim, b_dim, activation, normalization, bias=bias) for i in range(n_bottlenecks)])

    #return [encodings,latent]
    def encode(self,x):
        # Encoder
        x_encoder = []
        for block in self.encoder:
            x = block(x)
            x_encoder.append(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
        return x_encoder, x



# trial to preprocess the attribute (a few FC layers), not used in the end
def attribute_pre_treat(attr_dim,first_dim,max_dim,n_layers):
    #linear features for attributes
    layers = []
    in_channels = attr_dim
    out_channels = first_dim
    for i in range(n_layers):
        layers.append(nn.Sequential(nn.Linear(in_channels, out_channels),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        in_channels = out_channels
        out_channels=min(2*out_channels,max_dim)
    return nn.Sequential(*layers)





# used for the attribute: repeat spacially the attribute a and concat to feature
def reshape_and_concat(feat,a):
    a = a.unsqueeze(-1).unsqueeze(-1)
    attr = a.repeat((1,1, feat.size(2), feat.size(3)))
    return torch.cat([feat, attr], dim=1)

# decoder conv layers
def build_decoder_layers(conv_dim=64, n_layers=6, max_dim=512, im_channels=3, attr_dim=0,n_attr_deconv=0,activation='leaky_relu', normalization='batch', additional_channels=[], first_conv=False): 
    bias = normalization != 'batch'
    decoder = nn.ModuleList()
    shift = 0 if not first_conv else 1
    for i in reversed(range(shift,n_layers)): 
        #size of inputs/outputs
        dec_out = int(min(max_dim,conv_dim * 2 ** (i-1)))
        dec_in = min(max_dim,conv_dim * 2 ** (i))
        if i >= n_layers - n_attr_deconv: dec_in = dec_in + attr_dim #concatenate attribute
        if (i==shift): dec_out=conv_dim // 4 # last conv is of dim dim / 4
        if (i-shift < len(additional_channels)): dec_in += additional_channels[i-shift] # add some channels to last x layers

        dec_layer=[ConvReluBn(nn.Conv2d(dec_in, dec_out, 3, 1, 1,bias=bias,padding_mode='reflect'),activation=activation,normalization=normalization)]
        if (i==shift and len(additional_channels)>0):
            dec_layer+=[ConvReluBn(nn.Conv2d(dec_out, dec_out, 3, 1, 1,bias=bias,padding_mode='reflect'),activation,normalization)]
        decoder.append(nn.Sequential(*dec_layer))

    last_kernel= 3
    last_conv = nn.ConvTranspose2d(dec_out, im_channels, last_kernel, 1, last_kernel//2, bias=True) 
    return decoder, last_conv


# a simple unet (encoder decoder)
class Unet(nn.Module):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, normalization='instance', first_conv=False, n_bottlenecks=2, img_size=128, batch_size=32):
        super(Unet, self).__init__()
        self.n_layers = n_layers
        ##### build encoder
        self.encoder = Encoder(conv_dim,n_layers,max_dim,im_channels,normalization=normalization, first_conv=first_conv, n_bottlenecks=n_bottlenecks)
        ##### build decoder
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim, 3, first_conv=first_conv, normalization=normalization)


    #return [encodings,bneck]
    def encode(self,x):
        return self.encoder.encode(x)

    def decode(self, bneck):
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            out = resize_right.resize(out, scale_factors=2)
            out = dec_layer(out)
        x = self.last_conv(out) 
        x = torch.tanh(x)
        #x = x / torch.sqrt((x**2).sum(dim=1,keepdims=True))
        return x

    def forward(self, x):
        # propagate encoder layers
        _,z = self.encode(x)
        return self.decode(z)


# G1 without normals (similar to faderNet)
class MaterialEditGenerator(Unet):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, attr_dim=1,n_attr_deconv=1, normalization='instance', first_conv=False, n_bottlenecks=2, img_size=128, batch_size=32):
        super(MaterialEditGenerator, self).__init__(conv_dim, n_layers, max_dim, im_channels, normalization,first_conv,n_bottlenecks, img_size, batch_size)
        self.attr_dim = attr_dim
        self.n_attr_deconv = n_attr_deconv
        ##### change decoder : get attribute as input
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim,3, attr_dim=attr_dim, n_attr_deconv=n_attr_deconv,normalization=normalization, first_conv=first_conv)

    #adding the attribute if needed
    def add_attribute(self,i,out,a):
        if i < self.n_attr_deconv:
            out = reshape_and_concat(out, a)
        return out


    def decode(self, a, bneck):
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            out = resize_right.resize(out, scale_factors=2)
            out = self.add_attribute(i,out,a)
            out = dec_layer(out)
        x = self.last_conv(out) 
        x = torch.tanh(x)

        return x

    def forward(self, x,a):
        # propagate encoder layers
        _,z = self.encode(x)
        return self.decode(a,z)

# G1 (MaterialEditGenerator with normals)
class MaterialEditGeneratorWithNormals(MaterialEditGenerator):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, attr_dim=1,n_attr_deconv=1,n_concat_normals=1,normalization='instance', first_conv=False, n_bottlenecks=2, img_size=128, batch_size=32):
        super(MaterialEditGeneratorWithNormals, self).__init__(conv_dim, n_layers, max_dim, im_channels, attr_dim,n_attr_deconv,normalization,first_conv,n_bottlenecks, img_size, batch_size)
        self.n_concat_normals = n_concat_normals
        self.first_conv=first_conv
        self.pretreat_attr=False

        ##### change decoder : get normal as input, add some channels
        additional_channels=[3 for i in range(self.n_concat_normals)]
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim,3, attr_dim=attr_dim, n_attr_deconv=n_attr_deconv, additional_channels=additional_channels,normalization=normalization,first_conv=first_conv)
        # normal resizers
        device='cpu'
        antialiasing=True
        interpolation=interp_methods.cubic
        self.resize_normals=[]
        for n in range(n_concat_normals-1):
            self.resize_normals.append(resize_right.ResizeLayer([batch_size,3,img_size,img_size], scale_factors=0.5**(n+1), device=device, interp_method=interpolation, antialiasing=antialiasing))

        # unused but need to stay there because it is saved in snapshots
        self.attr_FC = attribute_pre_treat(attr_dim,32,32,2)

    def prepare_pyramid(self,map,n_levels):
        map_pyramid=[map]
        for i in range(n_levels-1):
            map_pyramid.insert(0,self.resize_normals[i](map))
        return map_pyramid


    #adding the normal map at the right scale if needed
    def add_multiscale_map(self,i,out,map_pyramid,n_levels):
        shift = 0 if not self.first_conv else 1 #PIX2PIX there is one layer less than n_layers
        rank=i-(self.n_layers-shift-n_levels)
        if rank >= 0 and rank<len(map_pyramid): 
            out = (torch.cat([out, map_pyramid[rank]], dim=1)) 
        return out

    def decode(self, a, bneck, normals):
        x,_ = self.decode_with_features(a, bneck, normals)
        return x

    # decode and return all the feature maps
    def decode_with_features(self, a, bneck, normals):
        features=[]
        #prepare normals
        normal_pyramid = self.prepare_pyramid(normals,self.n_concat_normals)
        #go through net
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            out = resize_right.resize(out, scale_factors=2)
            out = self.add_attribute(i,out,a)
            out = self.add_multiscale_map(i,out,normal_pyramid,self.n_concat_normals)
            out = dec_layer(out)
            features.append(out)
        x = self.last_conv(out)
        x = torch.tanh(x)
        return x,features

    def forward(self, x,a,normals):
        # propagate encoder layers
        _,z = self.encode(x)
        return self.decode(a,z,normals)


# G2 
class MaterialEditGeneratorWithNormals2Steps(MaterialEditGeneratorWithNormals):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, attr_dim=1,n_attr_deconv=1,n_concat_normals=1,normalization='instance', first_conv=False, n_bottlenecks=2, img_size=128, batch_size=32):
        super(MaterialEditGeneratorWithNormals2Steps, self).__init__(conv_dim, n_layers, max_dim, im_channels, attr_dim,n_attr_deconv,n_concat_normals,normalization,first_conv,n_bottlenecks, img_size, batch_size)
        self.img_size=img_size
        
        ##### change decoder : add the additional inputs from G1 + normals
        feat_channels=[8, 32, 64, 128, 256]
        if self.img_size>128: feat_channels.insert(0,0)
        additional_channels=[3+feat_channels[i] for i in range(self.n_concat_normals)]

        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim,3, attr_dim=attr_dim, n_attr_deconv=n_attr_deconv, additional_channels=additional_channels, normalization=normalization,first_conv=first_conv)

        # unused but need to stay there because it is saved in snapshots
        self.attr_FC = attribute_pre_treat(attr_dim,16,16,2)


    def decode(self, a, bneck, normals, g1_output):
        # prepapre illum and normals
        g1_pyramid = g1_output[-self.n_concat_normals:]
        normal_pyramid = self.prepare_pyramid(normals,self.n_concat_normals)
        #go through decoder
        out=bneck
        for i, dec_layer in enumerate(self.decoder):
            out = resize_right.resize(out, scale_factors=2)
            out = self.add_attribute(i,out,a)
            out = self.add_multiscale_map(i,out,normal_pyramid,self.n_concat_normals) # add normals
            out = self.add_multiscale_map(i+(1 if self.img_size>128 else 0),out,g1_pyramid,self.n_concat_normals) #add features
            out = dec_layer(out)
        x = self.last_conv(out)
        x = torch.tanh(x)

        return x

    def forward(self, x,a,normals, g1_output):
        # propagate encoder layers
        _, z = self.encode(x)
        return self.decode(a,z,normals, g1_output)



def FC_layers(in_dim,fc_dim,out_dim,tanh):
    layers=[nn.Linear(in_dim, fc_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Linear(fc_dim, out_dim)]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)



# LD
class Latent_Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels = 3,conv_dim=64, fc_dim=1024, n_layers=5,normalization='instance', first_conv=False):
        super(Latent_Discriminator, self).__init__()
        layers = []
        self.n_bnecks=3
        n_dis_layers = int(np.log2(image_size))
        layers=build_encoder_layers(conv_dim,n_dis_layers, max_dim, im_channels, normalization=normalization,activation='leaky_relu',dropout=0.3, first_conv=first_conv)

        self.conv = nn.Sequential(*layers[n_layers:])
        self.pool = nn.AvgPool2d(1 if not first_conv else 2)
        out_conv = min(max_dim,conv_dim * 2 ** (n_dis_layers - 1))
        self.fc_att = FC_layers(out_conv,fc_dim,attr_dim,True)

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_att


# classical GAN discriminator
class Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels = 3, conv_dim=64, fc_dim=1024, n_layers=5,normalization='instance'):
        super(Discriminator, self).__init__()
        layers = []
        layers=build_disc_layers(conv_dim,n_layers, max_dim, im_channels, normalization=normalization,activation='leaky_relu')
        self.conv = nn.Sequential(*layers)

        c_dim=min(max_dim,conv_dim * 2 ** (n_layers-1))
        self.last_conv = nn.Conv2d(c_dim, 1, 4, 1, 1) 

    def forward(self, x):
        y = self.conv(x)
        logit_real = self.last_conv(y)
        return logit_real


# discriminator that also tries to guess the attribute
class DiscriminatorWithClassifAttr(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels = 3, conv_dim=64, fc_dim=1024, n_layers=5,normalization='instance'):
        super(DiscriminatorWithClassifAttr, self).__init__()
        #convolutions for image
        layers=build_disc_layers(conv_dim,n_layers, max_dim, im_channels, normalization=normalization,activation='leaky_relu')
        self.conv_img = nn.Sequential(*layers)

        # branch for vanilla GAN
        c_dim=min(max_dim,conv_dim * 2 ** (n_layers-1))
        self.last_img_conv = nn.Conv2d(c_dim, 1, 4, 1, 1) #size of kernel: 1,3 or 4

        # branch for classification
        self.pool = nn.AvgPool2d(int(image_size/(2**n_layers)))
        self.fc_att = FC_layers(c_dim,fc_dim,attr_dim,True)

    def forward(self, x):
        img_feat = self.conv_img(x)
        # real/fake
        logit_real = self.last_img_conv(img_feat)
        # classif
        y = self.pool(img_feat)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_real, logit_att

