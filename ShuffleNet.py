import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

import random
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#load data
data = datasets.ImageFolder(root='./images', transform=transform)
data_loader = DataLoader(data, batch_size=32, shuffle=True)

#split data into training and testing sets
train_data, test_data = random_split(data, [0.8, 0.2])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

#import open source shufflenet model on PyTorch
model = torchvision.models.shufflenet_v2_x1_0(weights=None)

num_classes = len(data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 18

#training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
train_accs = []


for epoch in range(num_epochs):
    model.train()
    
    running_loss_train = 0.0
    correct_train = 0
    total_train = 0
    
    
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss_train += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        
    epoch_loss_train = running_loss_train / len(train_data)
    train_acc = correct_train / total_train
    train_losses.append(epoch_loss_train)
    train_accs.append(train_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss_train:.4f}")

#learning curve
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accs)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


torch.save(model.state_dict(), 'shufflenet_model.pth')


#classification report
state_dict = torch.load('./shufflenet_model.pth')
model.load_state_dict(state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

classification_rep = classification_report(all_labels, all_predictions, target_names=data.classes)
print("Classification Report:")
print(classification_rep)


running_loss_test = 0.0
correct_test = 0
total_test = 0
total_time = 0.0

test_losses = []
test_accs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        
        total_time += end_time - start_time
        
        loss = criterion(outputs, labels)
        test_losses.append(loss.item())
        
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        test_accs.append(accuracy)
        

avg_test_loss = sum(test_losses) / len(test_losses)
avg_test_acc = sum(test_accs) / len(test_accs)
avg_inference_time = total_time / len(test_loader)

print('\n'f"Average test loss: {avg_test_loss:.4f}\n")
print(f"Average test accuracy: {avg_test_acc:.4f}\n")
print(f"Average inference time: {avg_inference_time:.4f} seconds\n")

plt.figure()
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, len(test_accs) + 1), test_accs, label='Test Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.show()


correct_count, total_count = 0, 0

for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    
    idx = random.randint(0, len(images) - 1)
    sample_image = images[idx].unsqueeze(0)
    sample_label = labels[idx]

    with torch.no_grad():
        outputs = model(sample_image)
        _, predicted = torch.max(outputs, 1)
    
    predicted_label = predicted.item()
    actual_label = sample_label.item()
    correct = 'Correct' if predicted_label == actual_label else 'Incorrect'
    
    test_data_classes = data.classes
    
    plt.figure()
    sample_image = sample_image.cpu().squeeze().numpy()
    sample_image = np.transpose(sample_image, (1, 2, 0))
    sample_image = 0.5 * sample_image + 0.5  
    plt.title(f"Actual Label: {test_data_classes[actual_label]}, Predicted Label: {test_data_classes[predicted_label]} ({correct})")
    plt.imshow(sample_image)
    
    if predicted_label == actual_label:
        correct_count += 1
    total_count += 1

accuracy = correct_count / total_count * 100
print(f"Accuracy of randomised sampling tests: {accuracy:.2f}%")
    


