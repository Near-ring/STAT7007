#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:03:45 2024

@author: annie
"""

import torch
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

class ImageClassifier:
    def __init__(self, data_dir, transform, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,model=None,device='mps'):
        torch.manual_seed(42)
        self.data = datasets.ImageFolder(root=data_dir, transform=transform)
        self.train_data, self.test_data, self.val_data = random_split(self.data, [train_ratio, test_ratio, val_ratio])
        self.model = model
        self.device = device
    def training_model(self, model, dataset, epochs=10, batch_size=600, learning_rate=0.1, device='mps'):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        running_loss = 0.0

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() # * inputs.size(0)
                n_batches += 1
            epoch_loss = running_loss / n_batches
            accuracy = self.evaluate_model(model, dataloader)

            acc_tr.append(accuracy)
            epoch_losses.append(epoch_loss)
            print(f'Epoch [{i}/epoch], Loss: {epoch_loss:.4f} ,Test Accuracy: {accuracy:.4f}')

            # Save the trained model
            torch.save({'epoch': i+1,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'accuracy':accuracy}, 'squeezenet_model_'+str(i+1)+'_' + str(learning_rate)+'_'+str(batch_size)+'.pth')
        return acc_tr, epoch_losses

    def evaluate_model(self, model, dataloader , device='mps',classfication_report=False):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            #dataloader = DataLoader(testset,shuffle=False)
            #print(len(dataloader))
            for images, labels in dataloader:
                #print(images)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        
        if(classfication_report):
            classification_rep = classification_report(all_labels, all_preds, target_names=self.data.classes,zero_division=0)
            print("Classification Report:")
            print(classification_rep)



        return accuracy

