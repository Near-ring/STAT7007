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
from torchvision.models import SqueezeNet
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
    data_dir = 'data/images/images'
    classifier = ImageClassifier(data_dir, transform)
    device = torch.device('mps')
    model = models.squeezenet1_0(weights=None)
    model.to(device)
    epochs = 30
    batch_size = 32
    learning_rate = 0.001
    
    
    acc_tr, epoch_losses = classifier.training_model(model, classifier.train_data, epochs=epochs, batch_size=batch_size, 
                                                     learning_rate=learning_rate, device=device)
