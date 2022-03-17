# 
TODO : paths dataset/weights, titre, lien papier, verif requirements, lines to test, checkpoints normals
This repo contains the official code for the paper "" (CGF 2022). Project page:

## Requirements
- [Python 3.6+](https://www.python.org)
- [PyTorch 1.0+](https://pytorch.org)

- [tensorboardX 1.6+](https://github.com/lanpa/tensorboardX)
- [torchsummary](https://github.com/sksq96/pytorch-summary)
- [tqdm](https://github.com/tqdm/tqdm)
- [Pillow](https://github.com/python-pillow/Pillow)
- [easydict](https://github.com/makinacorpus/easydict)
- pytorch-lightning (for normal prediction)


## Testing

Download the trained networks for the attributes Glossy and Metallic and put them into `experiments/` (keeping the directory structure):

`test_network.py` allows to launch the network on one image and a given attribute (it also does the normal prediction)

Use:
`test_network.py INPUT_IMAGE ATTR_VAL ATTRIBUTE OUTPUT_IMAGE` edit the image INPUT_IMAGE with ATTRIBUTE set to ATTR_VAL. The trained weights should be in `experiments/final_step1_ATTRIBUTE/checkpoints/G_final.pth.tar` and `experiments/final_step2_ATTRIBUTE/checkpoints/G_final.pth.tar`.

Example use:
`test_network.py test_images/....png 1.0 glossy test_glossy_1.png`
`test_network.py test_images/....png 0.0 glossy test_glossy_0.png`
`test_network.py test_images/....png 1.0 metallic test_metallic_1.png`
`test_network.py test_images/....png 0.0 metallic test_metallic_0.png`

Note: It creates a temporary image test_normals.png in the current folder.


## Organization of the code

* `agents` : The main training files (architecture, optimisation scheme) are in `agents`. `trainingModule` is an abstract agent that regroup some basic training functions (similar to Pytorch lighting abstraction. Most of the optimization/training procedure is is `MaterialEditNet` (the most simple one), the others build on it by just changing a few key functions (group the inputs, define the nets):
  * `MaterialEditNet` : architecture of G1 without the normals, contains most of the code (equivalent to faderNet)
  * `MaterialEditNet_with_normals` : G1
  * `MaterialEditNet_with_normals_2steps` : G2
* `models` : code of the networks
* `datasets` : code to read the datasets
* `utils` : various utilities

Data:
* `configs` : configuration files to launch the trainings or test
* `experiments` : snapshots of experiments
* `test_images` : snapshots of experiments

Normal prediction:
* `pix2normal` : code of the network
* `normal_inference.py` to get the normal prediction for one image

## Training

Download the dataset:

The folder `configs` contains the configuration files used to train the network as in the paper.

Train G1 : `python train.py train_materialEditNet_withnormals.yaml`

Train G2 (after G1 is trained) : `python train.py train_materialEditNet_2steps.yaml`

Parameters that might require to be changed: `data_root` (path to dataset), `attributes` (attribute to edit), `g1_checkpoint` for G2 (path to trained G1), `checkpoint` (to load weights, for test or resuming training)
