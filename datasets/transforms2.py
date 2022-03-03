import numpy as np
from PIL import Image
import random

import torch
#from torchvision import transforms as T
#from torchvision.transforms import functional as F
from opencv_transforms import functional as F
from opencv_transforms import transforms as T
import albumentations as A
from albumentations import functional as FA



# def pad_if_smaller(img, size, fill=0):
#     min_size = min(img.size)
#     if min_size < size:
#         ow, oh = img.size
#         padh = size - oh if oh < size else 0
#         padw = size - ow if ow < size else 0
#         img = F.pad(img, (0, 0, padw, padh), fill=fill)
#     return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, normals):
        for t in self.transforms:
            image, normals = t(image, normals)
        return image, normals



class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, normals):
        image = F.center_crop(image, self.size)
        normals = F.center_crop(normals, self.size)
        return image, normals




class Resize(object): 
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = (size,size) #TODO according to what is size
        self.interpolation = interpolation

    def __call__(self, image, normals): #TODO warning for mask -> nearest interpolation?
        image = F.resize(image, self.size, self.interpolation)
        normals = F.resize(normals, self.size, self.interpolation)
        return image, normals


class ToTensor(object):
    def __call__(self, image, normals):
        image = F.to_tensor(image)
        normals = F.to_tensor(normals)
        #normals = torch.as_tensor(np.array(normals), dtype=torch.int64) #for masks
        return image, normals


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, normals):
        image = F.normalize(image, mean=self.mean, std=self.std)
        normals = F.normalize(normals, mean=self.mean, std=self.std)
        return image, normals


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            normals = F.hflip(normals) 
            normals_red_channel = normals[:,:,0]
            normals_red_channel = 255 - normals_red_channel
            normals[:,:,0] = normals_red_channel
        return image, normals


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            normals = F.vflip(normals)
            normals_green_channel = normals[:,:,1]
            normals_green_channel = 255 - normals_green_channel
            normals[:,:,1] = normals_green_channel
        return image, normals

class Random90DegRotClockWise(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            angle = 90
            image =  F.rotate(image, angle, resample=False, expand=False, center=None)
            normals =  F.rotate(normals, angle, resample=False, expand=False, center=None)

            normals_red_channel = normals[:,:,0]
            normals_green_channel = normals[:,:,1]

            normals_red_channel = 255 - normals_red_channel
            normals[:,:,0] = normals_green_channel
            normals[:,:,1] = normals_red_channel
        return image, normals

class Random180DegRot(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            angle = 180
            image =  F.rotate(image, angle, resample=False, expand=False, center=None)
            normals =  F.rotate(normals, angle, resample=False, expand=False, center=None)

            normals_red_channel = normals[:,:,0]
            normals_green_channel = normals[:,:,1]

            normals_red_channel = 255 - normals_red_channel
            normals_green_channel = 255 - normals_green_channel
            normals[:,:,0] = normals_red_channel
            normals[:,:,1] = normals_green_channel
        return image, normals

class Albumentations(object):
    def __init__(self, hue_limit, sat_limit, flip_prob):
        self.hue_limit = hue_limit
        self.sat_limit = sat_limit
        self.flip_prob=flip_prob

    def __call__(self, image, normals):
        if random.random() < self.flip_prob:
            hue_shift=random.uniform(-self.hue_limit, self.hue_limit)
            sat_shift=random.uniform(-self.sat_limit, self.sat_limit)
            image[:,:,:3] = FA.shift_hsv(image[:,:,:3], hue_shift=hue_shift,sat_shift=sat_shift, val_shift=0)
        return image, normals


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, normals):
        #image = pad_if_smaller(image, self.size)
        #normals = pad_if_smaller(normals, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        normals = F.crop(normals, *crop_params)
        return image, normals



class RandomResize(object):
    def __init__(self, low, high, interpolation=Image.BILINEAR):
        self.low = low
        self.high = high
        self.interpolation = interpolation

    def __call__(self, image, normals): 
        size = np.random.randint(self.low, self.high)
        image = F.resize(image, (size,size), self.interpolation)
        normals = F.resize(normals, (size,size), self.interpolation)
        return image, normals


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.degrees = degrees
        self.expand = expand
        self.resample = resample
        self.center = center
        self.fill = fill

    def __call__(self, image, normals): 
        angle = T.RandomRotation.get_params(self.degrees)
        image =  F.rotate(image, angle, self.resample, self.expand, self.center)
        normals =  F.rotate(normals, angle, self.resample, self.expand, self.center ) 
        #TODO change normals
        return image, normals

