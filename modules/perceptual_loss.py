import functools
import math
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as trf_F


def gram_matrix(input):
    # https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    a, b, c, d = input.size()
    N = a * b * c * d
    sqrt_N = math.sqrt(N)
    features = input.view(a * b, c * d).div(sqrt_N)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G  # normalize the values of the gram matrix


def random_crop_ixs(img, output_size):
    w, h = img.shape[-2:][::-1]
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    if w < tw:
        tw = w
    if h < th:
        th = h

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


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


class StyleLoss(nn.Module):
    def __init__(self, loss_fn=F.mse_loss):
        super(StyleLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, x_act, y_act):
        # x_act and y_act are dicts on the following form {layer_name: features}
        loss = 0
        for k in x_act.keys():
            g_x_act = gram_matrix(x_act[k])
            g_y_act = gram_matrix(y_act[k]).detach()
            loss += self.loss_fn(g_x_act, g_y_act)
        return loss / len(x_act)


if __name__ == '__main__':
    a = torch.tanh(torch.randn(4, 3, 256, 256))
    b = torch.tanh(torch.randn(4, 3, 256, 256))
    fext = VGG16FeatureExtractor(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    P_loss = PerceptualLoss()
    S_loss = StyleLoss()
    f_a = fext(a)
    f_b = fext(b)

    print(P_loss(f_a, f_b))
    print(S_loss(f_a, f_b))
