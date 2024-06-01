import os
import random
import shutil


dataset_root = './images_new'

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2



class_folders = os.listdir(dataset_root)

for class_folder in class_folders:
    class_path = os.path.join(dataset_root, class_folder)
    
    images = os.listdir(class_path)
    random.shuffle(images)
    
    
    num_images = len(images)
    num_train = int(train_ratio * num_images)
    num_val = int(val_ratio * num_images)
    num_test = num_images - num_train - num_val
    
    train_folder = os.path.join('train', class_folder)
    val_folder = os.path.join('val', class_folder)
    test_folder = os.path.join('test', class_folder)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    for i, image in enumerate(images):
        image_path = os.path.join(class_path, image)
        if i < num_train:
            shutil.copy(image_path, os.path.join(train_folder, image))
        elif i < num_train + num_val:
            shutil.copy(image_path, os.path.join(val_folder, image))
        else:
            shutil.copy(image_path, os.path.join(test_folder, image))
