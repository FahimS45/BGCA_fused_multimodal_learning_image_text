"""Dataset and data loading for vision model."""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from torchvision import transforms
from typing import List, Tuple
import random
import torch

class ImageDataset(Dataset):
    def __init__(self, images: List[Image.Image], labels: List[int], transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_and_preprocess_data(data_dir: str, classes: List[str], target_size: tuple = (224, 224)) -> Tuple[List[Image.Image], np.ndarray]:
    """Load and preprocess image data from directory."""
    X = []
    y = []

    # Loop through the classes (Non-Malignant and Malignant)
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        
        # Loop through the images in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Open and convert image to RGB
            img = Image.open(img_path).convert('RGB')
            
            # Resize the image to target_size
            img = img.resize(target_size)
            
            # Append the resized PIL Image object to X
            X.append(img)
            
            # Append the corresponding class label to y
            y.append(i)

    return X, np.array(y)

def apply_adasyn(X: List[Image.Image], y: np.ndarray) -> Tuple[List[Image.Image], np.ndarray]:
    """Apply ADASYN oversampling to balance the dataset."""
    # Convert PIL Images to numpy arrays
    X_arrays = [np.array(img) for img in X]
    
    # Get the shape of the images
    img_shape = X_arrays[0].shape
    
    # Reshape the arrays to 2D for ADASYN
    X_reshaped = np.array([x.flatten() for x in X_arrays])
    
    # Apply ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_reshaped, y)
    
    # Reshape back to original image shape
    X_balanced = [x.reshape(img_shape) for x in X_resampled]
    
    # Convert back to PIL Images
    X_balanced = [Image.fromarray(x.astype('uint8')) for x in X_balanced]
    
    return X_balanced, y_resampled

def split_data(X: List[Image.Image], y: np.ndarray, test_size: float, val_size: float, seed: int) -> Tuple:
    """Split data into train/val/test sets."""
    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, 
        stratify=y_train_val, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_transforms():
    """Get data transforms for training and validation/testing."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(60),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                      train_transform, val_test_transform, batch_size: int, seed: int):
    """Create PyTorch DataLoaders for training, validation, and testing."""
    
    def seed_worker(worker_id):
        np.random.seed(seed)
        random.seed(seed)
    
    # Apply ADASYN to training data
    X_balanced, y_balanced = apply_adasyn(X_train, y_train)
    
    # Create datasets
    train_dataset = ImageDataset(X_balanced, y_balanced, transform=train_transform)
    val_dataset = ImageDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = ImageDataset(X_test, y_test, transform=val_test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, worker_init_fn=seed_worker)

    return train_loader, val_loader, test_loader