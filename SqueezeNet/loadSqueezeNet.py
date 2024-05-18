#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:05:38 2024

@author: annie
"""
from torchvision import datasets, transforms, models
import os
import torch


def squeezenet_model(model_path):
    
    model = models.squeezenet1_0(weights=None)
    
    if os.path.exists(model_path):
         checkpoint = torch.load('model/squeezenet_model_38_0.001_32.pth')
         model.load_state_dict(checkpoint['model_state_dict'])
         epoch = checkpoint['epoch']
         loss = checkpoint['loss']
         
         print("epoch", epoch ,"loss",loss)
         
    return model
        
if __name__ == "__main__":    
    model = squeezenet_model('model/squeezenet_model_38_0.001_32.pth')
    print(model.cpu().state_dict())
    torch.save(model.cpu().state_dict()
               , '/Users/annie/Documents/GitHub/STAT7007/SqueezeNet/model/squeezenet_model_best.pth')
