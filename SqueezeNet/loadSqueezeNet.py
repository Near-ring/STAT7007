#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:05:38 2024

@author: annie
"""
from torchvision import datasets, transforms, models


def squeezenet_model(self, model_path):
    
    model = models.squeezenet1_0(weights=None)
    
    if os.path.exists(model_path):
         checkpoint = torch.load(model_path)
         model.load_state_dict(checkpoint['model_state_dict'])
         epoch = checkpoint['epoch']
         loss = checkpoint['loss']
         
         print("epoch", epoch ,"loss",loss)
         
    return model
            

    