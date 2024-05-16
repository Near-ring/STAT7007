import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image,ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from scipy import ndimage
from skimage.transform import resize


# Load pre-trained SqueezeNet model
model = models.squeezenet1_1(pretrained=True)
model.eval()

# Define the GradCAM class
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.grads = None
        self.hooks = []

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output
        
        def backward_hook(module, grad_in, grad_out):
            self.grads = grad_out[0]

        # Register hooks to the last Fire module's expand3x3 convolutional layer
        for name, module in self.model.named_modules():
            if name == 'features.12.expand3x3':
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        score = output[0, target_class]
        score.backward()

        # Get the weights by global pooling the gradients
        weights = torch.mean(self.grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze(0).detach().cpu().numpy()
        cam = np.maximum(cam, 0)  # ReLU to ensure non-negative values

        # Normalize the heatmap
        cam -= np.min(cam)
        cam /= np.max(cam)
                
        return cam
    
    
    def getbndBoxFromXML(self,bndbox):
        
        print(bndbox)
        xmin,ymin,xmax,ymax =0,0,0,0
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return xmin,ymin,xmax,ymax
            
          
    def plotImages(self, image_file, heatmap, label=None , xml_path=None , filename=None):
        
        #image = Image.open(image_path).convert('RGB')
        new_size = (2048, 1000)
        image_file = np.copy(image_file)
        image_file = resize(image_file, (image_file.shape[0], new_size[1], new_size[0]), anti_aliasing=True)
        heatmap = resize(heatmap, (heatmap.shape[0], new_size[1], new_size[0]), anti_aliasing=True)
        
        f, (ax1, ax2) = plt.subplots(2,1, layout='constrained')
        
        image_file= np.transpose(image_file, (1, 2, 0))
        image_file = 0.5 * image_file + 0.5
        rotated_image_file = image_file
        #rotated_image_file = ndimage.rotate(image_file, 0)
      
        ax1.imshow(image_file)
        
        if(xml_path !=None):
            if(os.path.exists(xml_path)):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bndboxes = obj.findall('bndbox')
                    for bndbox in bndboxes:
                        xmin,ymin,xmax,ymax = self.getbndBoxFromXML(bndbox)
                        width = xmax - xmin
                        height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                        ax1.add_patch(rect)
            
        # Add the patch to the Axes
        ax1.set_title("Original Image")    
        rotated_heatmap = np.transpose(heatmap, (1, 2, 0))

        ax2.imshow(image_file, alpha=1)
        ax2.imshow(rotated_heatmap, cmap='jet', alpha=0.5)
        ax2.set_title('GradCAM Heatmap')
        
        f.suptitle(str(label) , fontsize=16)
        
        ax1.axis('off')
        ax2.axis('off')
        
        if(filename!=None):       
            # Save the plot to a file
            output_file = '/Users/annie/Documents/GitHub/STAT7007/SqueezeNet/result/heat_map_result/'+filename+'.png' # Specify the output file name and format
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        
        plt.show()
        
        
    def plotImages1(self, image_file, heatmap, xml_path=None):
        
             
        f, (ax1, ax2) = plt.subplots(1,2, layout='constrained')
        ax1.imshow(image_file)
        
        if(os.path.exists(xml_path)):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                bndboxes = obj.findall('bndbox')
                print(bndboxes)
                for bndbox in bndboxes:
                    xmin,ymin,xmax,ymax = self.getbndBoxFromXML(bndbox)
                    print("xmax",xmax,xmin,ymin)
                    width = xmax - xmin
                    height = ymax - ymin
                    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax1.add_patch(rect)    # Add the patch to the Axes
        
        ax1.set_title('Original Image')
        
        ax2.imshow(heatmap.T, cmap='jet', alpha=0.5)
        ax2.set_title('GradCAM Heatmap')
        
        ax1.axis('off')
        ax2.axis('off')

        plt.show()
        
    def printOrginalImage(self, image_file,label, xml_path):
        
        #image = Image.open(image_path).convert('RGB')
       
        f, (ax1) = plt.subplots(1,1, layout='constrained')
        ax1.imshow(image_file)
        
        if(os.path.exists(xml_path)):
           tree = ET.parse(xml_path)
           root = tree.getroot()
           for obj in root.findall('object'):
               bndboxes = obj.findall('bndbox')              
               for bndbox in bndboxes:
                   xmin,ymin,xmax,ymax = self.getbndBoxFromXML(bndbox)
                   print("xmax",xmax,xmin,ymin)
                   width = xmax - xmin
                   height = ymax - ymin
                   rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                   ax1.add_patch(rect)    # Add the patch to the Axes
              
        # Add the patch to the Axes
        ax1.set_title('Original Image: ' + label)
        ax1.axis('off')

        plt.show()
        
        
   