import os
from PIL import Image
import random

def augment_image(image_path, save_dir):
    image = Image.open(image_path)
    
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    
    original_save_path = os.path.join(save_dir, f"{file_name}_original{file_extension}")
    image.save(original_save_path)
    
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_save_path = os.path.join(save_dir, f"{file_name}_flipped{file_extension}")
    flipped_image.save(flipped_save_path)
    
    
    vertical_flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    vertical_flipped_save_path = os.path.join(save_dir, f"{file_name}_vertical_flipped{file_extension}")
    vertical_flipped_image.save(vertical_flipped_save_path)
    
    rotated_image = image.transpose(Image.ROTATE_90)
    rotated_save_path = os.path.join(save_dir, f"{file_name}_rotated_90anticlock{file_extension}")
    rotated_image.save(rotated_save_path)
    
    rotated_image = image.transpose(Image.ROTATE_180)
    rotated_save_path = os.path.join(save_dir, f"{file_name}_rotated_180{file_extension}")
    rotated_image.save(rotated_save_path)
    
    rotated_image = image.transpose(Image.ROTATE_270)
    rotated_save_path = os.path.join(save_dir, f"{file_name}_rotated_90clock{file_extension}")
    rotated_image.save(rotated_save_path)
    
    rotated_image = image.transpose(Image.ROTATE_270)
    rotated_save_path = os.path.join(save_dir, f"{file_name}_rotated_90clock{file_extension}")
    rotated_image.save(rotated_save_path)
    
    angle = random.uniform(-10, 10)
    rotated_image_small_angle = image.rotate(angle, expand=True)
    rotated_save_path_small_angle = os.path.join(save_dir, f"{file_name}_randomly_rotated_{int(angle)}{file_extension}")
    rotated_image_small_angle.save(rotated_save_path_small_angle)
    

def augment_images_in_folder(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    
    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)
        
        if file_name.lower().endswith(('.jpg')):
            augment_image(file_path, target_folder)
            print(f"Processed {file_name}")

source_folder = "./images/waist folding"  
target_folder = "./images_new/waist folding"  
augment_images_in_folder(source_folder, target_folder)
