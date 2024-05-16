#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:04:24 2024

@author: annie
"""
import os
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
import utils_common
import time
import pandas as pd
import numpy as np
from squeezeNetGradCam import GradCAM

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    batch_sizes = [32,64,128]
    learning_rate = [0.0001,0.001,0.01,0.1]
    epoch = 100
    all_accuracy = []
    all_loss = []
       
    data_dir = 'data/images/images'
    classifier = ImageClassifier(data_dir, transform)
    model = models.squeezenet1_0()
    
    folders = sorted(os.listdir(data_dir))
    labels_folder = []
    for label, folder in enumerate(folders):
        labels_folder.append(folder)
        
    model_filename = 'squeezenet_model_38_0.001_32.pth'
    
    print("==========="+model_filename+"===========")
   
    #acc_list,loss_list,loss_images,loss_labels,pred_labels = classifier.report_one_model(model,'model/' +model_filename,'validation')

    
    # for i, images in enumerate(loss_images):

    #     gradcam = GradCAM(model)
    #     gradcam.register_hooks()
        
    #     heatmap_true = gradcam.generate(images, target_class=loss_labels[i])
    #     heatmap_pred = gradcam.generate(images, target_class=pred_labels[i])
        
    #     gradcam.remove_hooks()
        
    #     if loss_labels[i][0].cpu().numpy()==5 :
            
    #         true_label_index = loss_labels[i][0].cpu().numpy()
    #         pred_label_index = pred_labels[i][0].cpu().numpy()
            
    #         true_label  = labels_folder[true_label_index]
    #         pred_label = labels_folder[pred_label_index]
            
    #         gradcam.plotImages(images.squeeze().cpu().numpy(),heatmap_true,"true label " + true_label)
    #         gradcam.plotImages(images.squeeze().cpu().numpy(),heatmap_pred,"predicted Label " + pred_label)
    
    # for batch_size in batch_sizes:
    #     df_acc = pd.DataFrame()
    #     df_loss = pd.DataFrame()
        
    #     for lr in learning_rate:
    #         acc , loss = classifier.report(model,epoch,lr,batch_size,'validation')
    #         df_acc[str(lr)] = pd.Series(acc)
    #         df_loss[str(lr)]= pd.Series(loss)
        
    #     df_acc.to_csv('result/data_acc_batch'+str(batch_size)+'.csv', index=False)
    #     df_loss.to_csv('result/data_loss_batch'+str(batch_size)+'.csv', index=False)
      
    
    for batch_size in batch_sizes:  
        df_acc = pd.read_csv('result/data_acc_batch'+str(batch_size)+'.csv', header=0)
        df_loss= pd.read_csv('result/data_loss_batch'+str(batch_size)+'.csv', header=0)
        classifier.plot_cureve_csv(df_loss,df_acc, batch_size)
        for lr in learning_rate:
            if str(lr) in df_acc.columns.tolist():
                print("Accuracy : batch_sizes:",batch_size, ",learning rate: ",lr ,": " , df_acc[str(lr)].max(), "," ,  df_acc[str(lr)].idxmax()+1)
                print("Loss : batch_sizes:",batch_size, ",learning rate: ",lr ,": " , df_loss[str(lr)].min(), "," ,  df_loss[str(lr)].idxmin()+1)
    
    