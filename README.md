# Graphene-U-Net

# Abstract

Graphene-U-Net is a library that offers simple and easy to use functions for training and evaluating Neural Networks for Segmentation of microscopy images using the U-Net Architecture. The library is based on an implementation of U-Net in the PyTorch deep Llarning framework, and uses OpenCV/Numpy for the data handling as well as Scikit-learn for the evaluation metrics. It contains functions for loading the dataset, training using k fold cross validation, and inferring the network on new data.

# Authors

Robbie Sadre, Colin Ophus, Anastasiia Butko, Gunther Weber (Lawrence Berkeley National Laboratory, 1 Cyclotron Road, Berkeley, California 94720)

# License

See LICENSE.txt for licensing information..


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
