# ASAP-Net
This project implements **ASAP-Net** of paper ASAP-Net: Attention and Structure Aware Point Cloud Sequence Segmentation  **(BMVC2020)**.

![Semantic segmentation result on SemanticKITTI](https://github.com/intrepidChw/ASAP-Net/blob/master/pics/demo.gif)

## Overview

We improve spatio-temporal point cloud feature learning with a flexible module called
ASAP module considering both attention and structure information across frames, which can be combined with different backbones. Incorporating our module into backbones brings semantic segmentation performance improvements on both Synthia and SemanticKITTI datasets (**+3.4** to **+15.2** mIoU points with different backbones).

## Installation

The Synthia experiments is implemented with TensorFlow and the SemanticKITTI experiments is implemented with PyTorch. We tested the codes under TensorFlow 1.13.1 GPU version, PyTorch 1.1.0,  CUDA 10.0, g++ 5.4.0 and Python 3.6.9 on Ubuntu 16.04.12 with TITAN RTX GPU. **For SemanticKITTI experiments, you should  have a GPU memory of at least 16GB**.

#### Compile TF Operators for Synthia Experiments

We use the implementation in [xingyul](https://github.com/xingyul)/**[meteornet](https://github.com/xingyul/meteornet)**. Please follow the instructions below.

The TF operators are included under `Synthia_experiments/tf_ops`, you need to compile them first by `make` under each ops subfolder (check `Makefile`) or directly use the following commands:

```
cd Synthia_experiments
sh command_make.sh
```

 **Please update** `arch` **in the Makefiles for different** [CUDA Compute Capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) **that suits your GPU if necessary**.

#### Compile Torch Operators for SemanticKITTI Experiments

We use the PoinNet++ implementation in [sshaoshuai](https://github.com/sshaoshuai)/**[Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)**. Use the commands below to build Torch operators. 

```
cd SemanticKITTI_experiments/ASAP-Net_PointNet2/pointnet2
python setup.py install
```

## Experiments on Synthia

The codes for experiments on Synthia is in `Synthia_experiments/semantic_seg_synthia`. Please refer to `Synthia_experiments/semantic_seg_synthia/README.md` for more information on data preprocessing and running instructions.

## Experiments on SemanticKITTI

The **[SemanticKITTI_experiments/ImageSet2]https://github.com/intrepidChw/ASAP-Net/tree/master/SemanticKITTI_experiments/ImageSet2** folder contains dataset split information. Please put it under your semanticKITTI dataset like `Path to semanticKITTI dataset/dataset/sequences`.

#### PointNet++ as Backbone

The codes for framework with PointNet++ as Backbone is in `SemanticKITTI_experiments/ASAP-Net_PointNet2`. Please refer to `SemanticKITTI_experiments/ASAP-Net_PointNet2/README.md` for more information on running instructions.

#### SqueezeSegV2 as Backbone

The codes for framework with SqueezeSegV2 as Backbone is in `SemanticKITTI_experiments/ASAP-Net_SqueezeSegV2`. Please refer to `SemanticKITTI_experiments/ASAP-Net_SqueezeSegV2/README.md` for more information on running instructions.

## Acknowledgements

Special thanks for open source codes including  [xingyul](https://github.com/xingyul)/**[meteornet](https://github.com/xingyul/meteornet)**, [sshaoshuai](https://github.com/sshaoshuai)/**[Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)** and [PRBonn](https://github.com/PRBonn)/**[lidar-bonnetal](https://github.com/PRBonn/lidar-bonnetal)**.

## Citation

Please cite these papers in your publications if it helps your research:

```
@article{caoasap,
  title={ASAP-Net: Attention and Structure Aware Point Cloud Sequence Segmentation},
  author={Cao, Hanwen and Lu, Yongyi and Lu, Cewu and Pang, Bo and Liu, Gongshen and Yuille, Alan}
  booktitle={British Machine Vision Conference (BMVC)},
  year={2020}
}
```