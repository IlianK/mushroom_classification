import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
import random
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px


############ PATHS ############

SRC_DIR = Path.cwd()
ROOT_DIR = SRC_DIR.parent
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')

RAW_DIR = os.path.join(DATA_DIR, 'raw')
EXPLORATION_PATH = os.path.join(ROOT_DIR, 'exploration')
CSV_PATH = os.path.join(DATA_DIR, 'csv_mappings', 'train.csv')


############ DATA LOAD AND DISPLAY ############

def preload_images(data, dataset_dir):
    image_dict = {}
    for _, row in data.iterrows():
        img_name = f"{int(row['Image']):05}.jpg"
        img_path = os.path.join(dataset_dir, img_name)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            image_dict[img_name] = img
        else:
            print(f"Warning: Image {img_name} not found.")
    
    return image_dict


def get_image_filenames(mushroom_data, class_name, num_images=10, is_random=False):
    class_data = mushroom_data[mushroom_data['ClassName'] == class_name]
    image_filenames = class_data['Image'].astype(str).apply(lambda x: f"{x.zfill(5)}.jpg")

    if is_random:
        image_filenames = random.sample(list(image_filenames), min(len(image_filenames), num_images))
    else:
        image_filenames = image_filenames.head(num_images)
    
    return image_filenames


def display_images(image_filenames, image_dict):
    num_rows = (len(image_filenames) + 4) // 5
    fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    for i, filename in enumerate(image_filenames):
        image = image_dict.get(filename)
        
        if image:
            axes[i].imshow(image)
            axes[i].axis('off')
            axes[i].set_title(f"{filename}")
        else:
            axes[i].axis('off')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_images_per_class(mushroom_data, image_dict, class_name, num_images=10, is_random=False):
    image_filenames = get_image_filenames(mushroom_data, class_name, num_images, is_random)
    display_images(image_filenames, image_dict)


############ PLOT DISPLAY ############

def show_class_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='ClassName', order=data['ClassName'].value_counts().index)
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.show()


def plot_image_size_distribution(image_dict):
    widths = []
    heights = []
    
    for filename, image in image_dict.items():
        width, height = image.size
        widths.append(width)
        heights.append(height)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(widths, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Width Distribution')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(heights, bins=30, color='salmon', edgecolor='black')
    axes[1].set_title('Height Distribution')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def get_image_size_statistics(image_dict):
    widths = []
    heights = []
    
    for filename, image in image_dict.items():
        width, height = image.size
        widths.append(width)
        heights.append(height)
    
    width_mean = np.mean(widths)
    width_median = np.median(widths)
    width_std = np.std(widths)
    
    height_mean = np.mean(heights)
    height_median = np.median(heights)
    height_std = np.std(heights)
    
    stats = {
        'Statistic': ['Mean', 'Median', 'Standard Deviation'],
        'Width': [width_mean, width_median, width_std],
        'Height': [height_mean, height_median, height_std]
    }
    
    df_stats = pd.DataFrame(stats)
    return df_stats


def calculate_aspect_ratio(image_dict):
    aspect_ratios = []
    
    for image in image_dict.values():
        width, height = image.size
        aspect_ratio = width / height
        aspect_ratios.append(aspect_ratio)
    
    plt.hist(aspect_ratios, bins=30, color='green', edgecolor='black')
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (Width/Height)')
    plt.ylabel('Frequency')
    plt.show()


def plot_pixel_intensity_histogram(image_dict):
    reds, greens, blues = [], [], []
    
    for image in image_dict.values():
        image_np = np.array(image)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            r, g, b = image_np[..., 0].mean(), image_np[..., 1].mean(), image_np[..., 2].mean()
        else:
            r = g = b = image_np.mean()

        reds.append(r)
        greens.append(g)
        blues.append(b)
    
    plt.figure(figsize=(12, 6))
    plt.hist(reds, bins=30, color='red', alpha=0.5, label='Red')
    plt.hist(greens, bins=30, color='green', alpha=0.5, label='Green')
    plt.hist(blues, bins=30, color='blue', alpha=0.5, label='Blue')
    plt.title('Pixel Intensity Distribution by Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


############ CLUSTERING RESULTS ############

def load_results(method_name, path=EXPLORATION_PATH):
    result_csv_path = os.path.join(path, 'all_data', f'{method_name.lower()}_results.csv')
    result_df = pd.read_csv(result_csv_path)
    result = result_df.iloc[:, :-1].values 
    class_labels = result_df['Class'].values 
    return result, class_labels


def save_dimensionality_reduction_results(result, class_labels, method_name, path=EXPLORATION_PATH):
    os.makedirs(path, exist_ok=True)

    result_df = pd.DataFrame(result, columns=[f'{method_name} {i+1}' for i in range(result.shape[1])])
    result_df['Class'] = class_labels

    result_df.to_csv(os.path.join(path, f'{method_name.lower()}_results.csv'), index=False)
    np.save(os.path.join(path, f'{method_name.lower()}_result.npy'), result)

    if result.shape[1] == 2: 
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=result_df.iloc[:, 0], y=result_df.iloc[:, 1], hue=result_df['Class'], palette='Set2')
        plt.title(f'{method_name} Visualization')
        plt.xlabel(f'{method_name} 1')
        plt.ylabel(f'{method_name} 2')
        plt.savefig(os.path.join(path, f'{method_name.lower()}_visualization.png'))
        plt.close()

    print(f"{method_name} results saved in {path}")


def visualize_results(reduced_data, clusters, class_labels, technique="PCA", n_clusters=10, save_path=None):
    if reduced_data.shape[1] == 2:
        df = pd.DataFrame(reduced_data, columns=["Component 1", "Component 2"])
    elif reduced_data.shape[1] == 3:
        df = pd.DataFrame(reduced_data, columns=["Component 1", "Component 2", "Component 3"])
    else:
        raise ValueError("The reduced data must be 2D or 3D.")

    df["Cluster"] = clusters
    df["Class"] = class_labels  
    
    if reduced_data.shape[1] == 2:
        fig = px.scatter(df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Class"],
                         title=f'{technique} Clustering ({n_clusters} Clusters)', color_continuous_scale='Viridis')
    else:
        fig = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3", color="Cluster", hover_data=["Class"],
                            title=f'{technique} 3D Clustering ({n_clusters} Clusters)', color_continuous_scale='Viridis')
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{technique}_clustering_{n_clusters}_clusters.png")
        fig.write_image(file_path)
    else:
        fig.show()


############ CLUSTERING PREPERATION ############

def resize_and_flatten_images(image_dict, image_size=(224, 224)):
    flattened_images = []
    
    for image in image_dict.values():
        image_rgb = image.convert("RGB")
        image_resized = image_rgb.resize(image_size)
        flattened_images.append(np.array(image_resized).flatten())
    
    flattened_images_array = np.array(flattened_images)
    if len(set(flattened_images_array.shape[1:])) > 1:
        raise ValueError("Images are not of the same shape after resizing.")
    
    return flattened_images_array


def standardize_data(flattened_images):
    scaler = StandardScaler()
    scaled_images = scaler.fit_transform(flattened_images)
    return scaled_images


############ DIMENSIONALITY REDUCTION METHODS ############

def apply_pca(scaled_images, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_images)
    return pca_result, pca


def apply_tsne(scaled_images, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    tsne_result = tsne.fit_transform(scaled_images)
    return tsne_result, tsne


def apply_umap(scaled_images, n_components=2, n_neighbors=15, min_dist=0.1):
    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    umap_result = umap.fit_transform(scaled_images)
    return umap_result, umap


def apply_kmeans(data, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans


############ PER CLASS FUNCTIONS ############

def plot_pixel_intensity_per_class(image_dict, mushroom_data):
    reds, greens, blues, class_names = [], [], [], []
    
    for filename, image in image_dict.items():
        image_np = np.array(image)
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            r, g, b = image_np[..., 0].mean(), image_np[..., 1].mean(), image_np[..., 2].mean()
        else:
            r = g = b = image_np.mean()

        reds.append(r)
        greens.append(g)
        blues.append(b)

        class_name = mushroom_data[mushroom_data['Image'] == int(filename[:5])]['ClassName'].values[0]
        class_names.append(class_name)
    
    df = pd.DataFrame({
        'Red': reds,
        'Green': greens,
        'Blue': blues,
        'Class': class_names
    })

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Class', y='Red', palette='Reds', showfliers=False)
    plt.title('Red Channel Pixel Intensity per Class')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Class', y='Green', palette='Greens', showfliers=False)
    plt.title('Green Channel Pixel Intensity per Class')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Class', y='Blue', palette='Blues', showfliers=False)
    plt.title('Blue Channel Pixel Intensity per Class')
    plt.xticks(rotation=45)
    plt.show()


def cluster_per_class(mushroom_data, image_dict, class_labels, n_clusters=10, path=EXPLORATION_PATH):
    unique_classes = mushroom_data['ClassName'].unique()

    for class_name in unique_classes:
        # Filter data for the class
        class_data = mushroom_data[mushroom_data['ClassName'] == class_name]
        
        # Get image filenames for the class
        image_filenames = get_image_filenames(mushroom_data, class_name, num_images=len(class_data))
        class_image_dict = {img_name: image_dict[img_name] for img_name in image_filenames}
        
        # Flatten and scale images for clustering
        flattened_images = resize_and_flatten_images(class_image_dict)
        scaled_images = standardize_data(flattened_images)

        # Dimensionality Reduction (PCA, t-SNE, UMAP)
        pca_result, pca = apply_pca(scaled_images, n_components=2)
        tsne_result, tsne = apply_tsne(scaled_images, n_components=2, perplexity=30)
        umap_result, umap = apply_umap(scaled_images, n_components=2, n_neighbors=15, min_dist=0.1)

        # K-Means Clustering
        pca_clusters, pca_kmeans = apply_kmeans(pca_result, n_clusters=n_clusters)
        tsne_clusters, tsne_kmeans = apply_kmeans(tsne_result, n_clusters=n_clusters)
        umap_clusters, umap_kmeans = apply_kmeans(umap_result, n_clusters=n_clusters)

        # Save results per class
        save_dimensionality_reduction_results(pca_result, class_labels, 'PCA', path=os.path.join(path, class_name))
        save_dimensionality_reduction_results(tsne_result, class_labels, 't-SNE', path=os.path.join(path, class_name))
        save_dimensionality_reduction_results(umap_result, class_labels, 'UMAP', path=os.path.join(path, class_name))

        # Visualize results for each class
        visualize_results(pca_result, pca_clusters, class_labels, technique="PCA", n_clusters=n_clusters, save_path=os.path.join(path, class_name))
        visualize_results(tsne_result, tsne_clusters, class_labels, technique="t-SNE", n_clusters=n_clusters, save_path=os.path.join(path, class_name))
        visualize_results(umap_result, umap_clusters, class_labels, technique="UMAP", n_clusters=n_clusters, save_path=os.path.join(path, class_name))

        print(f"Clustering and results saved for class: {class_name}")