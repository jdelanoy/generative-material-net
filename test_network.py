import os, sys
from models.networks import MaterialEditGeneratorWithNormals, MaterialEditGeneratorWithNormals2Steps
import torch
import cv2
import numpy as np
import datasets.transforms2 as T
from utils import resize_right
from matplotlib import pyplot as plt
from utils.im_util import denorm_and_alpha
import torchvision.utils as tvutils

device=torch.device("cpu")


def load_model_from_path(model,path):
    checkpoint = torch.load(path,map_location=device)
    to_load = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(to_load)
    model.to(device)

def load_image_and_normals(input_img_path, normal_path):
    # read image
    image_rgb = cv2.cvtColor(cv2.imread(input_img_path, 1), cv2.COLOR_BGR2RGB)
    size=image_rgb.shape[0]
    # read normals 
    normals_bgra = cv2.imread(normal_path, -1)
    normals = np.ndarray((size,size,4), dtype=np.uint8)
    if normals_bgra.shape[0] != size :
            normals_bgra = cv2.resize(normals_bgra,(size,size))
    cv2.mixChannels([normals_bgra], [normals], [0,2, 1,1, 2,0, 3,3])

    #slighlty erode mask so that the results are nicer
    normals[:,:,3] = cv2.dilate(normals[:,:,3], cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    # add mask to image
    image = np.ndarray(normals.shape, dtype=np.uint8)
    cv2.mixChannels([image_rgb,normals], [image], [0,0, 1,1, 2,2, 6,3])

    transform = T.Compose([
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5,0), std=(0.5, 0.5, 0.5,1))
        ])
    image,normals = transform(image,normals) 
    # mask the normals and input image
    normals = normals*normals[3:]
    image = image*image[3:]

    return image[None], normals[None,:3]



input_img_path =sys.argv[1]
attribute_val = [float(sys.argv[2])]
attribute = sys.argv[3]
output_path = sys.argv[4]

############# predict normal map
normal_path = "test_normals.png"
os.system("python normal_inference.py "+input_img_path+" "+normal_path)


############# load the 2 networks + checkpoints
checkpoint_path_G1 = "experiments/final_step1_"+attribute+"/checkpoints/G_final.pth.tar"
G1 = MaterialEditGeneratorWithNormals(conv_dim=32, n_layers=6, max_dim=512, im_channels=4, attr_dim=1,n_attr_deconv=1,n_concat_normals=4,normalization='none', first_conv=False, n_bottlenecks=2, img_size=128)
load_model_from_path(G1,checkpoint_path_G1)
G1.eval()

checkpoint_path_G2 = "experiments/final_step2_"+attribute+"/checkpoints/G_final.pth.tar"
G2 = MaterialEditGeneratorWithNormals2Steps(conv_dim=32, n_layers=4, max_dim=512, im_channels=4, attr_dim=1,n_attr_deconv=1,n_concat_normals=4,normalization='none', first_conv=True, n_bottlenecks=3, img_size=256)
load_model_from_path(G2,checkpoint_path_G2)
G2.eval()

############# load images and prepare data
image, normals = load_image_and_normals(input_img_path, normal_path)
attribute_val = torch.FloatTensor(attribute_val)[None]

############# generate image
with torch.no_grad():
    rescaled_im=resize_right.resize(image, scale_factors=0.5)
    rescaled_normals=resize_right.resize(normals, scale_factors=0.5)
    _,z = G1.encode(rescaled_im)
    g1_output, g1_features = G1.decode_with_features(attribute_val,z,rescaled_normals)

    output_g2 = G2.forward(image, attribute_val, normals, g1_features)
    tvutils.save_image(denorm_and_alpha(output_g2,image[:,3:])[0],output_path)

