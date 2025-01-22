import os
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt
from torchvision import transforms

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.load() 
            return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_and_process_image(image_path):
    img = load_image(image_path)
    if img is None:
        return None
    if img.mode != 'RGB':  
        img = img.convert('RGB')  
    return train_transforms(img)


def preprocess_samples(n, raw_dir):
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
        processed_img = load_and_process_image(image_path)
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
            
            axes[original_idx].imshow(original_images[i])
            axes[original_idx].set_title(f'Original {i + 1}')
            
            processed_img = processed_images[i].permute(1, 2, 0).numpy()
            processed_img = (processed_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            processed_img = processed_img.clip(0, 1)
            
            axes[original_idx + len(axes) // 2].imshow(processed_img)
            axes[original_idx + len(axes) // 2].set_title(f'Processed {i + 1}')
            
            if not axis_on:
                axes[original_idx].axis('off')
                axes[original_idx + len(axes) // 2].axis('off')
        
        plt.tight_layout()
        plt.show()


def verify_preprocessing_sample(n, raw_dir, axis_on=False):
    original_images, processed_images = preprocess_samples(n, raw_dir)
    show_images(original_images, processed_images, axis_on)


def preprocess_and_save_as_tensors(raw_dir, preprocessed_dir):
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir) 

    image_files = [f for f in os.listdir(raw_dir) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(raw_dir, image_file)
        try:
            tensor_img = load_and_process_image(image_path)
            if tensor_img is not None:
                tensor_path = os.path.join(preprocessed_dir, image_file.replace('.jpg', '.pt'))
                torch.save(tensor_img, tensor_path)
                print(f"Processed and saved: {tensor_path}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
