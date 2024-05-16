#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:04:24 2024

@author: annie
"""

import torch
from time import time
import torch.optim as optim
from torchvision.io import read_image
from torchvision import datasets ,transforms ,models 
from torchvision.models import SqueezeNet
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from squeezeNet_imageClassifier import ImageClassifier
import matplotlib.pyplot as plt
import utils_common
import time
import pandas as pd
import numpy as np
from squeezeNetGranCam import GradCAM
from PIL import Image,ImageDraw
import os
import random
if __name__ == "__main__":
    
    model = models.squeezenet1_0(weights=None)       
    
    for iter in range(38):
        
        checkpoint = torch.load('model/squeezenet_model_'+str(iter)+'_0.001_32.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
        # Load and preprocess image
        
        #image_path = "data/images/images/crease/img_01_4402117100_00006.jpg"
        # xml_path = "data/label/label/img_01_425501700_00022.xml"
        
        data_dir = "/Users/annie/Documents/GitHub/STAT7007/SqueezeNet/data/images/images"
        xml_dir = "/Users/annie/Documents/GitHub/STAT7007/SqueezeNet/data/label/label/"
        
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
        
        dataloader = datasets.ImageFolder(root=data_dir, transform=transform)
        
        # for images, labels in dataloader:
     
        #     input_image = images.unsqueeze(0)
        
        #     # Instantiate GradCAM and generate heatmap
        #     gradcam = GradCAM(model)
        #     gradcam.register_hooks()
            
        #     # Assuming you want to target a specific class, e.g., class 0
        #     heatmap = gradcam.generate(input_image, target_class=labels)
        #     gradcam.remove_hooks()
            
        #     gradcam.plotImages(input_image.squeeze().cpu().numpy(),heatmap)
            
        all_data=[]
        
        # List all folders in the data directory
        folders = sorted(os.listdir(data_dir))
        labels_folder = []
        
        for label, folder in enumerate(folders):
            #print(folder,label)
            labels_folder.append(folder)
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
               # List all image files in the folder
               images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
               #random.shuffle(images)
               xmls = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
               all_data.extend([(os.path.join(folder_path, image), label ,os.path.splitext(image)[0]) for image in images ])
        
        classArray = [[] for _ in range(10)]  # Assuming there are 10 classes
        for data in all_data:
            image_path, image_class_label, filename = data
            classArray[image_class_label].append(data)
            
        for i, class_list in enumerate(classArray): 
            for data in classArray[i][:1]:
                image_path , image_class_label , filename = data
                #get label ==5
                if(image_class_label!=None):
                    image = Image.open(image_path).convert('RGB')
                    input_image = transform(image)
                    input_image= input_image.unsqueeze(0)
                    
                    xml_filepath = xml_dir + filename + '.xml'
                    
                    # Instantiate GradCAM and generate heatmap
                    gradcam = GradCAM(model)
                    gradcam.register_hooks()
                    
                    # Assuming you want to target a specific class, e.g., class 0
                    heatmap = gradcam.generate(input_image, target_class=image_class_label)
                    gradcam.remove_hooks()            
                    gradcam.plotImages(input_image.squeeze().cpu().numpy(),heatmap,labels_folder[image_class_label] , xml_filepath , str(image_class_label)+"/" + filename +'_iter_'+str(iter))
                    #gradcam.plotImages(input_image.squeeze().cpu().numpy(),heatmap,labels_folder[image_class_label] , xml_filepath )
                    
                    #gradcam.printOrginalImage(image,labels_folder[image_class_label],xml_filepath)
