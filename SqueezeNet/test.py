#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:04:24 2024

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
import matplotlib.pyplot as plt



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    batch_size = 32
    learning_rate = 0.001
    epoch = 30
    
    acc = []
    epochs = []
    
    for i in range(epoch):
    
        device = torch.device('mps')
        model = models.squeezenet1_0(weights=None)
        checkpoint = torch.load('squeezenet_model_'+str(i+1)+'_'+ str(learning_rate) +'_' + str(batch_size)+'.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        #accuracy = checkpoint['accuracy']
    
        print("epoch", epoch ,"loss",loss)
        model.to(device)
        
        model.eval()
        
        data_dir = 'data/images/images'
        classifier = ImageClassifier(data_dir, transform)
    
        #trainDataloader = DataLoader(classifier.train_data,shuffle=False)
        validationDataloader = DataLoader(classifier.val_data,shuffle=False)
        #testDataloader = DataLoader(classifier.test_data,shuffle=False)
        
        #trainacc = classifier.evaluate_model(model,trainDataloader)
        valacc = classifier.evaluate_model(model,validationDataloader,classfication_report=False)
        #testacc = classifier.evaluate_model(model,testDataloader)
        
        #print("acc",testacc ,"valacc",valacc , "trainacc",trainacc)
        print("valacc",valacc )
        acc.append(valacc)
        epochs.append(i)
        

    # Adding labels and title
    plt.plot(epochs,acc)
    plt.xlabel('Validation Accuracy')
    plt.ylabel('epochs')
    plt.title('Validation accuracy Chart')
    
    # Displaying the plot
    plt.show()
            
    