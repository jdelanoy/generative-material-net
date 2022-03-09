import torch
import torch.nn as nn
import torch.nn.functional as F


# generator class from https://github.com/znxlwm/pytorch-pix2pix/blob/master/network.py
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size,stride, mid_channels=None,up=False,extra_padding=0,downsample_opt=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.downsample_opt = downsample_opt
        self.stride = stride
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1,padding_mode = 'reflect',bias=False),
            nn.BatchNorm2d(mid_channels,momentum=0.9),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=2+extra_padding,padding_mode = 'reflect',bias=False),
            nn.BatchNorm2d(out_channels,momentum=0.9)
        )
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=stride,padding=0,padding_mode = 'reflect',bias=False),
                nn.BatchNorm2d(out_channels,momentum=0.9),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        input = x
        out = self.double_conv(input)
        if self.stride>1 or (self.in_channels>self.out_channels) or (self.downsample_opt):
            input = self.downsample(x)
        out+=input
        out = F.leaky_relu(out, 0.2)

        if self.up==True:
            out = self.upsample(out)
        return out

class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()

        # Unet encoder
        self.conv0 = ResBlock(4, d, 3, 1,extra_padding=-1,downsample_opt=True)
        self.conv1 = ResBlock(d, d, 4, 2)
        self.conv2 = ResBlock(d, d * 2, 4, 2)
        self.conv3 = ResBlock(d * 2, d * 4, 4, 2)
        self.conv4 = ResBlock(d * 4, d * 8, 4, 2)
        #self.conv5 = ResBlock(d * 4, d * 8, 3, 1,extra_padding=-1,downsample_opt=True)

        # Unet decoder
        #self.upsample1 = ResBlock(d * 8, d * 4,4,1)
        self.upsample2 = ResBlock(d * 4 * 2, d * 4, 3,1,up=True,extra_padding=-1)
        self.upsample3 = ResBlock(d * 4 * 2, d * 2,3,1,up=True,extra_padding=-1)
        self.upsample4 = ResBlock(d * 2 * 2, d,3,1,up=True,extra_padding=-1)
        self.upsample5 = ResBlock(d * 1* 2, d,3,1,up=True,extra_padding=-1)
        self.resblock6 = ResBlock(d,d,3,1,extra_padding=-1)
        self.deconv6 = nn.Conv2d(d, 3, kernel_size=3, padding=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        e0 = self.conv0(input)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1) 
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        #e5 = self.conv5(e4)

        #d1 = self.upsample1(e4)
        #d1 = torch.cat([d1, e4], 1)       
        #d2 = F.dropout(self.upsample2(d1), 0.2, training=True)                                            
        d2 = F.dropout(self.upsample2(e4), 0.2, training=True)
        d2 = torch.cat([d2, e3], 1)
        d3 = F.dropout(self.upsample3(d2), 0.2, training=True)
        d3 = torch.cat([d3, e2], 1)
        d4 = F.dropout(self.upsample4(d3), 0.2, training=True)
        d4 = torch.cat([d4, e1], 1)
        d5 = self.resblock6(self.upsample5(d4))
        d5 = self.deconv6(d5)
        o = torch.tanh(d5)
        o = o / torch.sqrt((o**2).sum(dim=1,keepdims=True))

        return o

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input[:,:3,:,:], label[:,:3,:,:]], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



