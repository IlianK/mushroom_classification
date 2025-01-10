import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import torch

############ LOAD AND PREPROCESS IMAGE DATA ###############

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.load() 
            return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return img


def load_and_process_image(image_path, target_size=(224, 224)):
    img = load_image(image_path)
    if img.mode != 'RGB':  
        img = img.convert('RGB')  
    img = preprocess_image(img, target_size)
    return img


def preprocess_samples(n, raw_dir, target_size=(224, 224)):
    image_files = [f for f in os.listdir(raw_dir) if f.endswith('.jpg')]
    sample_images = random.sample(image_files, n)

    original_images = []
    processed_images = []

    for image_file in sample_images:
        image_path = os.path.join(raw_dir, image_file)
        img = load_image(image_path)

        if img is None:
            print(f"Skipping {image_file} due to loading failure.")
            continue

        original_images.append(img) 
        processed_img = load_and_process_image(image_path, target_size)
        processed_images.append(processed_img)  

    return original_images, processed_images


def show_images(original_images, processed_images, axis_on=False):
    n = len(original_images)
    groups = (n + 4) // 5  
    
    for group in range(groups):
        start_idx = group * 5
        end_idx = min((group + 1) * 5, n)

        print(f"Showing group {group + 1} (Images {start_idx + 1} to {end_idx})")

        fig, axes = plt.subplots(2, end_idx - start_idx, figsize=(15, 3))
        axes = axes.flatten()

        for i in range(start_idx, end_idx):
            original_idx = i - start_idx 
            
            # Show original 
            axes[original_idx].imshow(original_images[i])
            axes[original_idx].set_title(f'Original {i + 1}')
            
            # Show processed
            axes[original_idx + len(axes) // 2].imshow(processed_images[i])
            axes[original_idx + len(axes) // 2].set_title(f'Processed {i + 1}')
            
            if not axis_on:
                axes[original_idx].axis('off')
                axes[original_idx + len(axes) // 2].axis('off')
        
        plt.tight_layout()
        plt.show()


def verify_preprocessing_sample(n, raw_dir, target_size=(224, 224), axis_on=False):
    original_images, processed_images = preprocess_samples(n, raw_dir, target_size)
    show_images(original_images, processed_images, axis_on)


def preprocess_and_save_as_tensors(raw_dir, preprocessed_dir, target_size=(224, 224)):
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir) 

    image_files = [f for f in os.listdir(raw_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(raw_dir, image_file)

        try:
            # Preprocess & convert to tensor
            img = load_and_process_image(image_path, target_size) 
            tensor_img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) 

            # Save as a .pt
            tensor_path = os.path.join(preprocessed_dir, image_file.replace('.jpg', '.pt'))
            torch.save(tensor_img, tensor_path)
            print(f"Processed and saved as tensor: {tensor_path}")

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
