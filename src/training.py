import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import display

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from torchvision import models, transforms
from torchvision.models import AlexNet_Weights

from tqdm import tqdm


############## PATHS ##################

SRC_DIR = Path.cwd()
ROOT_DIR = SRC_DIR.parent

DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
PREPROCESSED_DIR = os.path.join(DATA_DIR, 'preprocessed')
CSV_PATH = os.path.join(DATA_DIR, 'csv_mappings', 'train.csv')

MODEL_DIR = os.path.join(ROOT_DIR, 'models')
BASELINE_DIR = os.path.join(MODEL_DIR, 'baselines_finetuned')


############### DATA FUNCTIONALITY ###############

class MushroomDataset(Dataset):
    def __init__(self, preprocessed_dir, csv_path, transform=None):
        self.preprocessed_dir = preprocessed_dir  
        self.csv_path = csv_path  
        self.transform = transform  
        self.csv_data = pd.read_csv(csv_path)
        
        # Images and Labels
        self.image_ids = self.csv_data['Image'].values  
        self.labels = self.csv_data['Mushroom'].values 
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]

        image_id_str = str(image_id).zfill(5)  # Pad for filename
        
        # Load .pt files
        image_path = os.path.join(self.preprocessed_dir, f"{image_id_str}.pt")
        image = torch.load(image_path)  
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(preprocessed_dir, csv_path, batch_size=32):
    # Init
    dataset = MushroomDataset(preprocessed_dir, csv_path)

    # Split 
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    # Subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Dataloaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


############### MODEL FUNCTIONALITY ###############

def save_model(model, optimizer, epoch, loss, accuracy, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }

    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")


def train_on_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, scheduler):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # Training loop
    for batch_idx, data in enumerate(tqdm(train_loader, desc="[Train]")):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log loss and accuracy per batch to TensorBoard
        batch_accuracy = 100.0 * correct / total
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Train/Accuracy', batch_accuracy, epoch * len(train_loader) + batch_idx)

    train_accuracy = 100.0 * correct / total
    avg_train_loss = train_loss / len(train_loader)

    # Log the learning rate
    for param_group in optimizer.param_groups:
        writer.add_scalar('Train/Learning Rate', param_group['lr'], epoch)

    # Step the scheduler
    if scheduler:
        scheduler.step()

    return avg_train_loss, train_accuracy


def validate_on_epoch(model, val_loader, criterion, optimizer, device, epoch, writer, best_val_loss, patience, epochs_no_improve, save_path):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc="[Val]")):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log loss and accuracy per batch to TensorBoard
            batch_accuracy = 100.0 * correct / total
            writer.add_scalar('Validation/Loss', loss.item(), epoch * len(val_loader) + batch_idx)
            writer.add_scalar('Validation/Accuracy', batch_accuracy, epoch * len(val_loader) + batch_idx)

    val_accuracy = 100.0 * correct / total
    avg_val_loss = val_loss / len(val_loader)

    # Early stopping 
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        save_model(model, optimizer, epoch, avg_val_loss, val_accuracy, save_path)
        print(f"Model saved to {save_path}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        return best_val_loss, epochs_no_improve, True, val_accuracy 

    return best_val_loss, epochs_no_improve, False, val_accuracy


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, device, writer, scheduler, patience, save_path):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss, train_accuracy = train_on_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, scheduler)
        print(f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.2f}%")

        # Validate
        best_val_loss, epochs_no_improve, early_stop, val_accuracy = validate_on_epoch(
            model, val_loader, criterion, optimizer, device, epoch, writer, best_val_loss, patience, epochs_no_improve, save_path
        )
        print(f"Val Loss = {best_val_loss:.4f}, Val Acc = {val_accuracy:.2f}%")

        if early_stop:
            break  

    return model


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="[Test]"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store preds and labels for plots
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100.0 * correct / total
    avg_test_loss = test_loss / len(test_loader)

    print(f"Test Loss = {avg_test_loss:.4f}")
    print(f"Test Accuracy = {test_accuracy:.2f}%")
    
    return avg_test_loss, test_accuracy, all_labels, all_predictions


############### PRETRAINED MODELS ###############

def get_alexnet_model(num_classes, device):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)    
    return model

def get_resnet_model(num_classes, device):
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)    
    return model

def get_vgg16_model(num_classes, device):
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device) 
    return model

def get_densenet_model(num_classes, device):
    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)
    return model

def get_efficientnet_model(num_classes, device):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)
    return model


############### CUSTOM MODELS ################

class EnhancedResNet(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super(EnhancedResNet, self).__init__()

        self.resnet = models.resnet50(pretrained=True)  
        self.resnet.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.fc1 = nn.Linear(2048, 1024)  
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)  
        x = self.dropout(x) 
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x)  
        return x
    

class EnhancedAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # Layer 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),             # Layer 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),         # Layer 3
            nn.BatchNorm2d(192),                       # BatchNorm added
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),             # Layer 4
            nn.Conv2d(192, 384, kernel_size=3, padding=1),        # Layer 5
            nn.BatchNorm2d(384),                       # BatchNorm added
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),        # Layer 6
            nn.BatchNorm2d(256),                       # BatchNorm added
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),        # Layer 7
            nn.BatchNorm2d(256),                       # BatchNorm added
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),             # Layer 8
        )

        # Pooling layer to ensure consistent output size
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Classifier layers with Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),                    # Dropout 1
            nn.Linear(256 * 6 * 6, 4096),            # Fully Connected Layer 1
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                    # Dropout 2
            nn.Linear(4096, 4096),                  # Fully Connected Layer 2
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),               # Fully Connected Layer 3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = self.classifier(x)
        return x


def get_custom_alexnet(num_classes, device):
    model = EnhancedAlexNet(num_classes)
    model = model.to(device)
    return model

def get_custom_resnet(num_classes, device):
    model = EnhancedResNet(num_classes)
    model = model.to(device)
    return model


########### MODEL HELPERS ##############
def get_optimizer_criterion_scheduler(
    model, train_loader, epochs, lr=0.001, weight_decay=1e-5, momentum=0.9, scheduler_type="StepLR"
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    if scheduler_type == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )

    elif scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

    elif scheduler_type == "None":
        scheduler = None  
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return optimizer, criterion, scheduler


def set_model_for_training(model_type, train_loader, epochs, learning_rate, num_classes, device, scheduler_type="StepLR"):
    base_log_path = os.path.join(BASELINE_DIR, model_type, 'log')
    base_result_path = os.path.join(BASELINE_DIR, model_type, 'results')

    if os.path.exists(base_log_path):
        shutil.rmtree(base_log_path)
    os.makedirs(base_log_path, exist_ok=True)
    os.makedirs(base_result_path, exist_ok=True)

    if model_type == 'alexnet':
        model = get_alexnet_model(num_classes, device)

    elif model_type == 'resnet':
        model = get_resnet_model(num_classes, device)

    elif model_type == 'vgg16':
        model = get_vgg16_model(num_classes, device)

    elif model_type == 'densenet':
        model = get_densenet_model(num_classes, device)

    elif model_type == 'efficientnet':
        model = get_efficientnet_model(num_classes, device)

    elif model_type == 'custom_alexnet':
        model = get_custom_alexnet(num_classes, device)

    elif model_type == 'custom_resnet':
        model = get_custom_resnet(num_classes, device)

    else:
        raise ValueError(f"Unsupported model type")
    
    optimizer, criterion, scheduler = get_optimizer_criterion_scheduler(
        model, train_loader, epochs, lr=learning_rate, scheduler_type=scheduler_type
    )
    
    writer = SummaryWriter(log_dir=base_log_path)
    print(model)
    
    return model, optimizer, criterion, scheduler, base_result_path, writer


############### EVAL PLOTS ###############

def plot_confusion_matrix(all_labels, all_predictions, num_classes, save_path=None):
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=np.arange(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save_path:
        save_path = os.path.join(save_path, 'plot_confusion_matrix.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def per_class_accuracy(all_labels, all_predictions, num_classes, class_names, save_path=None):
    class_accuracies = []
    print('Class accuracies:')

    for i in range(num_classes):
        class_indices = [j for j, label in enumerate(all_labels) if label == i]
        class_predictions = [all_predictions[j] for j in class_indices]
        class_labels = [all_labels[j] for j in class_indices]
        
        class_accuracy = accuracy_score(class_labels, class_predictions)
        class_accuracies.append(class_accuracy)
        
        print(f"({i:<2}) {class_names[i]:<20}: {class_accuracy:.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_classes), class_accuracies, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(num_classes), labels=class_names if class_names else range(num_classes))
    plt.ylim(0, 1)

    if save_path:
        save_path = os.path.join(save_path, 'plot_per_class_accuracy.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Per-class accuracy plot saved to {save_path}")
  
    plt.show()

    return class_accuracies


def display_classification_report(all_labels, all_predictions):
    report_dict = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(len(set(all_labels)))], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    display(report_df)
    return report_df


def save_evaluation_results(save_path, avg_test_loss, test_accuracy, all_labels, all_predictions, per_class_acc, num_classes):
    save_path = os.path.join(save_path, 'evaluation_results.txt')
    with open(save_path, 'w') as f:
        f.write(f"Test Loss = {avg_test_loss:.4f}\n")
        f.write(f"Test Accuracy = {test_accuracy:.2f}%\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(num_classes)]))
        f.write("\nPer-Class Accuracy:\n")
        for i, acc in enumerate(per_class_acc):
            f.write(f"Class {i}: {acc:.2f}\n")