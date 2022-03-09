import torch
import datasets.transforms2 as T
import os
#import model
import pix2normal.pix2normal_module as pix2normal_module
import cv2
import numpy as np

from PIL import Image
import sys
from utils.im_util import denorm_and_alpha
import torchvision.utils as tvutils


if __name__ == "__main__":
    INPUT_HEIGHT = 128
    INPUT_WIDTH = 128
    INPUT_CHANNELS = 4
    NUM_INIT_FILTERS = 64
    MODEL_PATH = 'pix2normal/checkpoints//normal_final.ckpt'

    val_trf = T.Compose([
        T.Resize(128),
        T.ToTensor(),
    ])

    input_image=sys.argv[1]
    output_normal=sys.argv[2]
#imgs, filenames = load_images_from_folder(DATA_PATH)
    img = cv2.imread(input_image,cv2.IMREAD_UNCHANGED)

    model = pix2normal_module.pix2normal(NUM_INIT_FILTERS,
                                INPUT_CHANNELS,
                                INPUT_HEIGHT,
                                INPUT_WIDTH)

    checkpoint = torch.load(MODEL_PATH,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.freeze()

    
    image,__ = val_trf(img,img)
    normal_pred = model(image[None])
    print(image.shape, normal_pred.shape)
    tvutils.save_image(denorm_and_alpha(normal_pred,image[None,3:])[0],output_normal)


