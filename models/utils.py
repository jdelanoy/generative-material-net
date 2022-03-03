import torch.nn as nn


def activation_func(activation):
    assert activation in ['relu', 'leaky_relu', 'selu', 'none']
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    if activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    if activation == 'selu':
        return nn.SELU(inplace=True)
    if activation == 'none':
        return nn.Identity()


def normalization_func(normalization):
    assert normalization in ['batch', 'instance', 'none']

    if normalization == 'batch':
        return lambda ch: nn.BatchNorm2d(ch)
    if normalization == 'instance':
        return lambda ch: nn.InstanceNorm2d(ch)
    if normalization == 'none':
        return lambda _: nn.Identity()
