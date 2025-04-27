##dataset.py

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from config import batch_size, test_size, val_size, random_state, img_resize_x, img_resize_y

class AlzheimersDataset(Dataset):
    """
    Dataset class for Alzheimer's disease classification with 4 classes:
    - Class 0: Non-Demented (Normal)
    - Class 1: Very Mild Dementia
    - Class 2: Mild Dementia
    - Class 3: Moderate Dementia
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of paths to the MRI images
            labels (list): List of labels corresponding to the images
            transform: PyTorch transforms for the images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Class mapping for reference
        self.class_names = {
            0: "non_demented",
            1: "very_mild_dementia", 
            2: "mild_dementia",
            3: "moderate_dementia"
        }
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label


def calculate_mean_std(image_paths, batch_size=32):
    """
    Calculate the mean and standard deviation of the dataset.
    
    Args:
        image_paths (list): List of paths to the images
        batch_size (int): Batch size for processing
        
    Returns:
        mean, std: Lists containing channel-wise mean and standard deviation
    """
    temp_transforms = transforms.Compose([
        transforms.Resize((img_resize_x, img_resize_y)),
        transforms.ToTensor()
    ])
    
    temp_dataset = AlzheimersDataset(image_paths, [0] * len(image_paths), transform=temp_transforms)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    num_batches = 0
    
    print("Calculating dataset mean and std...")
    for data, _ in temp_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sq_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = torch.sqrt(channels_sq_sum / num_batches - mean**2)
    
    print(f"Calculated Mean: {mean.tolist()}")
    print(f"Calculated Std: {std.tolist()}")
    
    return mean.tolist(), std.tolist()


def prepare_data_loaders(data_dir, batch_size=32, test_size=0.2, val_size=0.2, random_state=42):
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for the data loaders
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random state for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader: PyTorch data loaders
    """
    # Classes and their corresponding label indices
    classes = {
        "non_demented": 0,
        "very_mild_dementia": 1,
        "mild_dementia": 2, 
        "moderate_dementia": 3
    }
    
    # Lists to store image paths and labels
    all_image_paths = []
    all_labels = []
    
    # Gather image paths and labels
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                all_image_paths.append(img_path)
                all_labels.append(label)
    
    # Split data into train+val and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    # Split train+val into train and val sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size, random_state=random_state, stratify=train_val_labels
    )
    
    # Calculate dataset-specific mean and std from training images
    dataset_mean, dataset_std = calculate_mean_std(train_paths, batch_size)
    
    # Define transforms with calculated normalization parameters
    train_transforms = transforms.Compose([
        transforms.Resize((img_resize_x, img_resize_y)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((img_resize_x, img_resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    
    # Create datasets
    train_dataset = AlzheimersDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = AlzheimersDataset(val_paths, val_labels, transform=test_transforms)
    test_dataset = AlzheimersDataset(test_paths, test_labels, transform=test_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Print dataset statistics
    print(f"Dataset statistics:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Testing: {len(test_dataset)} images")
    
    # Print class distribution
    class_counts = {class_name: train_labels.count(label) for class_name, label in classes.items()}
    print(f"Class distribution in training set:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    return train_loader, val_loader, test_loader

# Function to be imported in interface.py
def create_alzheimer_data_loaders(data_dir):
    """Helper function to create data loaders with config parameters"""
    return prepare_data_loaders(
        data_dir, 
        batch_size=batch_size, 
        test_size=test_size, 
        val_size=val_size, 
        random_state=random_state
    )
def get_normalization_parameters(data_dir):
    """Get dataset-specific mean and std values
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        mean, std: Lists containing channel-wise mean and standard deviation
    """
    # Get all images from the training set folder
    image_paths = []
    for class_name in ["non_demented", "very_mild_dementia", "mild_dementia", "moderate_dementia"]:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
    
    # Calculate and return mean and std
    return calculate_mean_std(image_paths)