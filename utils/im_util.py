import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import transform
import numpy as np
from torchvision import transforms
import cv2
import torch
from PIL import Image
import operator, itertools
from utils.batch_transforms import Normalize



def _imscatter(x, y, image, color=None, ax=None, zoom=1.):
    """ Auxiliary function to plot an image in the location [x, y]
        image should be an np.array in the form H*W*3 for RGB
    """
    if ax is None:
        ax = plt.gca()
    try:
        image=image.numpy().transpose((1,2,0))*0.5+0.5
        #image = plt.imread(image)
        size = min(image.shape[0], image.shape[1])
        image = transform.resize(image[:size, :size], (256, 256))
    except TypeError:
        # Likely already an array...
        pass
    #print(x)
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        edgecolor = dict(boxstyle='round,pad=0.05',
                         edgecolor=color, lw=1) \
            if color is not None else None
        ab = AnnotationBbox(im, (x0, y0),
                            xycoords='data',
                            frameon=True,
                            bboxprops=edgecolor,
                            )
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def denorm(x, device):
    #get from [-1,1] to [0,1]
    if x.size(1) == 4:
        norm=Normalize(mean=(-1,-1,-1,0), std=(2,2,2,1), device=device)
    else:
        norm=Normalize(mean=(-1,-1,-1), std=(2,2,2), device=device)
    return norm(x)
    #out = (x + 1) / 2
    #return out.clamp_(0, 1)

def write_labels_on_images(images, labels):
    for im in range(images.shape[0]):
        text_image=np.zeros((images.shape[2],images.shape[3],images.shape[1]), np.uint8)
        for i in range(labels.shape[1]):
            cv2.putText(text_image, "%.2f"%(labels[im][i].item()), (10,14*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255,255), 2, 8)
        image_numpy=((text_image.astype(np.float32))/255).transpose(2,0,1)+images[im].cpu().detach().numpy()
        images[im]= torch.from_numpy(image_numpy)


def get_alpha_channel(image): 
    image = image.convert('RGBA')
    alpha = image.getchannel('A')
    return alpha


def denorm_and_alpha(image,alpha=None):
    image=denorm(image,device=image.device)
    if alpha != None:
        return torch.cat([image*alpha,alpha],dim=1)
    return image