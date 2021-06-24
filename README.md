# Graphene-U-Net

## Abstract

Graphene-U-Net is a library that offers simple and easy to use functions for training and evaluating Neural Networks for Segmentation of microscopy images using the U-Net Architecture. The library is based on an implementation of U-Net in the PyTorch deep Llarning framework, and uses OpenCV/Numpy for the data handling as well as Scikit-learn for the evaluation metrics. It contains functions for loading the dataset, training using k fold cross validation, and inferring the network on new data.

## Authors

Robbie Sadre, Colin Ophus, Anastasiia Butko, Gunther Weber (Lawrence Berkeley National Laboratory, 1 Cyclotron Road, Berkeley, California 94720)

## License

See LICENSE.txt for licensing information.


## Copyright

Graphene U-Net Copyright (c) 2021, The Regents of the University of 
California, through Lawrence Berkeley National Laboratory (subject to 
receipt of any required approvals from the U.S. Dept. of Energy). 
All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Dept.
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

## Dependencies

This code has been verified using:

  * Python 3.7.6 / 3.8.5 
  * PyTorch-gpu 1.4.0  / 1.7.1
  * OpenCV 4.4.0  / 4.5.1
  * scikit-learn 0.22.1  / 0.23.2
  * NumPy 1.18.1  / 1.19.2
  * Pandas 1.0.0  / 1.1.5

This code uses [jvanvugt/pytorch-unet](https://github.com/jvanvugt/pytorch-unet/), copyright (c) 2018 Joris. 

# Getting Started

## Installation

```
git clone https://github.com/lbnlcomputerarch/graphene-u-net.git
```

## Usage

In terminal:
```
cd graphene-u-net/ 
jupyter notebook usage.ipynb  
```
## Publication

If you use this for research, please cite the original paper:

> Sadre, R., Ophus, C., Butko, A., & Weber, G. (2021). Deep Learning Segmentation of Complex Features in Atomic-Resolution Phase-Contrast Transmission Electron Microscopy Images. Microscopy and Microanalysis, 1-11. doi:10.1017/S1431927621000167


```
@article{sadre_ophus_butko_weber_2021, title={Deep Learning Segmentation of Complex Features in Atomic-Resolution Phase-Contrast Transmission Electron Microscopy Images}, DOI={10.1017/S1431927621000167}, journal={Microscopy and Microanalysis}, publisher={Cambridge University Press}, author={Sadre, Robbie and Ophus, Colin and Butko, Anastasiia and Weber, Gunther H.}, year={2021}, pages={1–11}}

```
