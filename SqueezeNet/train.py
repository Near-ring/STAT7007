#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:32:28 2024

@author: annie
"""
import torch
from time import time
import torch.optim as optim
from torchvision import datasets ,transforms ,models
#from torchvision.models import SqueezeNet
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from squeezeNet_imageClassifier import ImageClassifier



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    data_dir = '/Users/annie/Documents/GitHub/STAT7007/SqueezeNet/data/images/images'
    squeezeNetModel = models.squeezenet1_0(weights=None)
    classifier = ImageClassifier(data_dir, transform,model=squeezeNetModel)
   
    
    epochs = 100
    batch_size = 128
    learning_rate = [0.0001,0.001]
    
    for lr in learning_rate :
        acc_tr, epoch_losses = classifier.training_model(epochs=epochs, batch_size=batch_size, 
                                                         learning_rate=lr)


