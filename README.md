# 



## Requirements
- [Python 3.6+](https://www.python.org)
- [PyTorch 1.0+](https://pytorch.org)
- [tensorboardX 1.6+](https://github.com/lanpa/tensorboardX)
- [torchsummary](https://github.com/sksq96/pytorch-summary)
- [tqdm](https://github.com/tqdm/tqdm)
- [Pillow](https://github.com/python-pillow/Pillow)
- [easydict](https://github.com/makinacorpus/easydict)



## Organization of the code

* `agents` : The main training files (architecture, optimisation scheme) are in `agents`. `trainingModule` is an abstract agent that regroup some basic training functions (similar to Pytorch lighting abstraction. Most of the optimization/training procedure is is `MaterialEditNet` (the most simple one), the others build on it by just changing a few key functions (group the inputs, define the nets):
  * `MaterialEditNet` : architecture of G1 without the normals, contains most of the code (equivalent to faderNet)
  * `MaterialEditNet_with_normals` : G1
  * `MaterialEditNet_with_normals_2steps` : G2
* `configs` : configuration files to launch the trainings or test
* `datasets` : code to read the datasets
* `experiments` : snapshots of experiments
* `models` : code of the networks
* `utils` : various utilities


## Training
