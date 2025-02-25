import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import random_split

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import time

import random
import numpy as np
from tqdm import tqdm


def load_images(images_path: str, train_ratio=0.7, test_ratio=0.2, batch_size: int = 32,
                shuffle: bool = True, augment: bool = False):
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    augmented_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    torch.manual_seed(42)

    dataset = datasets.ImageFolder(root=images_path, transform=base_transform)

    # Split dataset into train and test
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Apply augmentations to the training dataset
    augmented_train_dataset = datasets.ImageFolder(root=images_path, transform=augmented_transform)

    # Combine original and augmented training datasets
    combined_train_dataset = ConcatDataset([train_dataset, augmented_train_dataset])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    train_loader_aug = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    if augment:
        return dataset, train_loader_aug, test_loader
    else:
        return dataset, train_loader, test_loader


def train_model(model, train_loader, optimizer: torch.optim, num_epochs):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    print(f"Using device: {device}")

    vec_train_loss = []
    vec_train_acc = []

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0

        start_time = time.time()
        for images, labels in tqdm(train_loader):
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
        end_time = time.time()
        duration = end_time - start_time
        est_time_sec = duration * (num_epochs - epoch)
        est_time_min = est_time_sec / 60

        epoch_loss_train = running_loss_train / total_train
        train_acc = correct_train / total_train
        vec_train_loss.append(epoch_loss_train)
        vec_train_acc.append(train_acc)
        print(f"\nEpoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss_train:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Estimated time remaining: {est_time_min:.2f} minutes\n")

    return vec_train_loss, vec_train_acc


def plot_train_curve(vec_train_loss, vec_train_acc):
    num_epochs = len(vec_train_loss)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), vec_train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), vec_train_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test_report(model, model_state_dict_path: str, test_loader: DataLoader, data: datasets.ImageFolder):
    state_dict = torch.load(model_state_dict_path)
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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


def eval_model(model, test_loader: DataLoader):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    test_losses = []
    test_accs = []
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            test_accs.append(accuracy)
    end_time = time.time()
    total_time = end_time - start_time

    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_acc = sum(test_accs) / len(test_accs)
    total_test_time = total_time

    print('\n'f"Average test loss: {avg_test_loss:.4f}\n")
    print(f"Average test accuracy: {avg_test_acc:.4f}\n")
    print(f"total_test_time: {total_test_time:.4f} seconds\n")
