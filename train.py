# Train network with dataset

'''
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

'''
import argparse

import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json

import time
#import helper
from PIL import Image

from image_classifier_utils import load_data, read_json
from model_check_point import Network, validation, train_NN, save_the_checkpoint


# Retreiving the command line inputs
parser = argparse.ArgumentParser(description="Train model")

#save checkponts
parser.add_argument("--arch", default="densenet169", help="choose model architecture")

#data directory
parser.add_argument("--data_dir", default='flowers/', help="load flowers data directory for training")

#set hyperparams
parser.add_argument("--learning_rate", type=int, default=0.001, help="choose architecture learning rate , 0.01")

parser.add_argument("--save_checkpoint", default ='checkpoint.pth' ,help="save the trained model")
parser.add_argument("--hidden_units", type=int, default=1024, help="architecture hidden units")
parser.add_argument("--epochs", type=int, default=4, help="architecture numer of epocs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="set the gpu mode")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")

parse_result = parser.parse_args()
#print(parse_result.learning_rate)

cat_to_name = read_json(parse_result.category_names)

trainloader, testloader, trainset = load_data(parse_result.data_dir)

model = train_NN(n_hidden=[parse_result.hidden_units],  n_epoch=parse_result.epochs, labelsdictionary=cat_to_name, lr=parse_result.learning_rate, \
            device=parse_result.gpu, model_name=parse_result.arch, trainloader=trainloader, trainset=trainset)



#print(parse_result.save_checkpoint)

save_the_checkpoint(model, parse_result.save_checkpoint)
