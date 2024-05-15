#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:03:45 2024

@author: annie
"""

import torch
import os
import torch.nn as nn
from tqdm import tqdm
from time import time
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import SqueezeNet
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool#
from squeezeNetGranCam import GradCAM

class ImageClassifier:
    
    def __init__(self, data_dir,  transform, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,model=None):
    
        torch.manual_seed(42)
        self.data = datasets.ImageFolder(root=data_dir, transform=transform)
        self.train_data, self.test_data, self.val_data = random_split(self.data, [train_ratio, test_ratio, val_ratio])
        self.model = model
        self.activation = {}
        
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        self.device = device
        
    def getFeatures(self):
        #for squeezeNet
        return_nodes = {
            # node_name: user-specified key for output dict
            'layer1.2.relu_2': 'layer1',
            'layer2.3.relu_2': 'layer2',
            'layer3.5.relu_2': 'layer3',
            'layer4.2.relu_2': 'layer4',
        }
        
        train_nodes, eval_nodes = get_graph_node_names(self.model)
        print(train_nodes)
        
    def training_model(self, epochs=10, batch_size=600, learning_rate=0.1):
        
        device = self.device
        self.model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        running_loss = 0.0

        dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        loop = tqdm(range(epochs), ncols=110)
        acc_tr = []
        epoch_losses = []

        for i in loop:
            t0 = time()
            epoch_loss = 0
            running_loss = 0
            n_batches = 0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() # * inputs.size(0)
                n_batches += 1
            epoch_loss = running_loss / n_batches
            accuracy, _ , loss_images , loss_labels,pred_labels = self.evaluate_model(self.model, dataloader)
 
            acc_tr.append(accuracy)
            epoch_losses.append(epoch_loss)
            print(f'Epoch [{i}/epoch], Loss: {epoch_loss:.4f} ,Test Accuracy: {accuracy:.4f}')

            # Save the trained model
            torch.save({'epoch': i+1,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'accuracy':accuracy}, '/Users/annie/Documents/GitHub/STAT7007/SqueezeNet/model/squeezenet_model_'+str(i+1)+'_' + str(learning_rate)+'_'+str(batch_size)+'.pth')
        return acc_tr, epoch_losses

    def evaluate_model(self, model, dataloader , device='mps',classfication_report=False):
        
        model.to(device)
        model.eval()
        all_preds = []
        all_labels = []
        all_losses = []
        loss_image = []
        loss_label = []
        pre_label = []
        criterion = nn.CrossEntropyLoss()
        
        start_time = time()
        with torch.no_grad():
            #dataloader = DataLoader(testset,shuffle=False)
            #print(len(dataloader))
            for images, labels in dataloader:
                #print(images)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                all_losses.append(loss.item())
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if(preds.cpu().numpy()!=labels.cpu().numpy()):
                    loss_image.append(images)
                    loss_label.append(labels)
                    pre_label.append(preds)
                    
                
        end_time = time()
        total_time = end_time - start_time
        accuracy = accuracy_score(all_labels, all_preds)
        
        avg_loss = sum(all_losses) / len(all_losses)
        
        print('\n'f"Average loss: {avg_loss:.4f}")
        print(f"Average accuracy: {accuracy:.4f}")
        print(f"total_time: {total_time:.4f} seconds\n")
        
        
        if(classfication_report):
            classification_rep = classification_report(all_labels, all_preds, target_names=self.data.classes,zero_division=0)
            print("Classification Report:")
            print(classification_rep)
            
        
        #if(printMissclassified):
       
        #self.gradCamPlot(loss_image, loss_label, model)

        return accuracy, avg_loss , loss_image, loss_label ,pre_label
    
    def gradCamPlot(self,loss_image,loss_label,model):
        
        model.eval()
        
        gradcam = GradCAM(model)
        gradcam.register_hooks()

        # Assuming you want to target a specific class, e.g., class 0]
        for i in range(len(loss_image)):
            image = loss_image[i]
            print("loss_label",loss_label[i].cpu())
            heatmap = gradcam.generate(image ,target_class=loss_label[i].cpu())
            
            gradcam.plotImages(image.squeeze().cpu().numpy(),heatmap)
            gradcam.remove_hooks()
        #print(image.squeeze(0).permute(1, 2, 0).shape)
        #print("classifier image:--------" , image.squeeze().cpu().numpy().shape)
       
    
    
    def plot_curve(self,vec_train_loss, vec_train_acc ,batch_size,learning_rate):
        
        # Add main title
         
        num_epochs = len(vec_train_loss)
        plt.figure()
        plt.subplot(1, 2, 1)
        for i , loss in enumerate(vec_train_loss):
            num_epochs = len(loss)
            print(i,learning_rate[i])
            plt.plot(range(1, num_epochs + 1), loss,label=str(learning_rate[i]))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(' Loss')
        plt.legend()
    
        plt.subplot(1, 2, 2)
        for i, acc in enumerate(vec_train_acc):
            num_epochs = len(acc)
            plt.plot(range(1, num_epochs + 1), acc,label=str(learning_rate[i]))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
    
        plt.suptitle("Batch size " + str(batch_size))
        plt.tight_layout()
        plt.show()

    
    def plot_cureve_csv(self, loss_df, accuracy_df, batch_size):
        
        # Plot Loss DataFrame
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        loss_df.plot(ax=plt.gca())  # Use current axis
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy DataFrame
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
        accuracy_df.plot(ax=plt.gca())  # Use current axis
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.suptitle("Batch size " + str(batch_size))
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
    
    def report(self, model, epoch ,learning_rate,batch_size, mode='validation'):
     
        acc_list = []
        epochs = []
        loss_list = []
        
        print("epoch" , epoch)
        for i in range(epoch):
            
            file_path = 'model/squeezenet_model_'+str(i+1)+'_'+ str(learning_rate) +'_' + str(batch_size)+'.pth'
            if os.path.exists(file_path):
                checkpoint = torch.load(file_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                #accuracy = checkpoint['accuracy']
            
                print("epoch", epoch ,"loss",loss)
                
                model.eval()
                
                if(mode=='train'):
                    dataloader = DataLoader(self.train_data,shuffle=False)  
                if(mode=='validation'):
                    dataloader = DataLoader(self.val_data,shuffle=False)
                if(mode=='test'):
                    dataloader = DataLoader(self.test_data,shuffle=False)
                
                acc ,loss ,_,_,_= self.evaluate_model(model,dataloader,classfication_report=False)
              
                acc_list.append(acc)
                epochs.append(i)
                loss_list.append(loss)
        
        return acc_list,loss_list
    
    
    def report_one_model(self, model, file_path, mode='validation'):
    
       acc_list = []
       loss_list = []
       
       #file_path = 'model/squeezenet_model_'+str(i+1)+'_'+ str(learning_rate) +'_' + str(batch_size)+'.pth'
       if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            #accuracy = checkpoint['accuracy']
            
            print("epoch", epoch ,"loss",loss)
            
            model.eval()
            
            if(mode=='train'):
                dataloader = DataLoader(self.train_data,shuffle=False)  
            if(mode=='validation'):
                dataloader = DataLoader(self.val_data,shuffle=False)
            if(mode=='test'):
                dataloader = DataLoader(self.test_data,shuffle=False)
            
            print(mode)
            acc ,loss ,loss_images, loss_labels ,pred_labels= self.evaluate_model(model,dataloader,classfication_report=True)
              
            acc_list.append(acc)
            loss_list.append(loss)
           
           
       return acc_list,loss_list,loss_images, loss_labels,pred_labels
               
   
       