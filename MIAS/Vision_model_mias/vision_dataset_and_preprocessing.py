"""Dataset and data loading for MIAS classification."""

import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from typing import List, Tuple
from collections import Counter
import random
import torch


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure image is RGB
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.shape[-1] == 3:
            image = Image.fromarray(image, mode="RGB")
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        if self.transform:
            image = self.transform(image)

        return image, label


def preprocess_image(img):
    """Preprocess a mammogram image to isolate the breast region."""
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    morphed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary_img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    isolated_breast = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_breast = isolated_breast[y:y+h, x:x+w]
    return cropped_breast


def apply_bicubic_interpolation(image, scale_factor=2, target_size=(224, 224)):
    """Enhance an image using bicubic interpolation-based super-resolution."""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    high_res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    high_res_image = cv2.resize(high_res_image, target_size)
    high_res_image = np.expand_dims(high_res_image, axis=-1)
    return high_res_image


def read_images(data_dir: str):
    """Read mammogram images from directory."""
    print("Reading images...")
    info = {}
    
    for i in range(322):
        if i < 9:
            image_name = f'mdb00{i + 1}'
        elif i < 99:
            image_name = f'mdb0{i + 1}'
        else:
            image_name = f'mdb{i + 1}'
        
        image_address = os.path.join(data_dir, f"{image_name}.pgm")
        img = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            info[image_name] = img
        else:
            print(f"Warning: Image {image_name} not found.")
    
    print(f"Total images read: {len(info)}")
    return info


def read_labels(data_dir: str):
    """Read labels from Info.txt file."""
    print("Reading labels...")
    filename = os.path.join(data_dir, 'Info.txt')
    with open(filename) as f:
        text_all = f.read()
    
    lines = text_all.split('\n')
    info = {}
    
    for line in lines:
        words = line.split(' ')
        if len(words) > 3:
            if words[3] == 'B':  # Benign
                info[words[0]] = 0
            elif words[3] == 'M':  # Malignant
                info[words[0]] = 1

    # Remove metadata entry if exists
    if 'Truth-Data:' in info:
        del info['Truth-Data:']
    
    return info


def augment_minority_classes(X_rgb, Y, train_transform):
    """Augment minority classes to balance dataset."""
    class_counts = Counter(Y)
    max_count = max(class_counts.values())
    
    augmented_images = []
    augmented_labels = []

    for label, count in class_counts.items():
        if count < max_count:
            minority_indices = np.where(Y == label)[0]
            images_to_augment = X_rgb[minority_indices]

            while count < max_count:
                for img in images_to_augment:
                    if count >= max_count:
                        break
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    augmented_img = train_transform(pil_img)
                    augmented_images.append(
                        np.array(augmented_img.permute(1, 2, 0))
                    )
                    augmented_labels.append(label)
                    count += 1

    if augmented_images:
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        X_rgb = np.concatenate((X_rgb, augmented_images), axis=0)
        Y = np.concatenate((Y, augmented_labels), axis=0)

    return X_rgb, Y


def load_and_preprocess_data(config):
    """Load and preprocess mammography data."""
    # Read images and labels
    image_info = read_images(config.data_dir)
    label_info = read_labels(config.data_dir)
    
    print(f"Total number of labels: {len(label_info)}")
    
    X, Y = [], []
    missing_labels = []

    # Process images
    for image_id in image_info.keys():
        # Apply preprocessing
        preprocessed_image = preprocess_image(image_info[image_id])
        # Apply bicubic interpolation
        high_res_image = apply_bicubic_interpolation(
            preprocessed_image, 
            scale_factor=config.scale_factor,
            target_size=config.target_size
        )
        
        X.append(high_res_image)
        
        if image_id in label_info:
            Y.append(label_info[image_id])
        else:
            missing_labels.append(image_id)
            Y.append(0)  # Default to benign

    X = np.array(X)
    Y = np.array(Y)
    
    # Convert grayscale to RGB
    X_rgb = np.stack([cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB) for img in X])
    
    print(f"X shape: {X_rgb.shape}")
    print(f"Y shape: {Y.shape}")
    if missing_labels:
        print(f"Images without labels: {missing_labels}")
    
    return X_rgb, Y


def split_data(X, y, test_size: float, val_size: float, seed: int):
    """Split data into train/val/test sets."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, 
        stratify=y_train_val, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_transforms():
    """Get data transforms for training and validation/testing."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=60, translate=(0.1, 0.1), 
                               scale=(0.8, 1.2), shear=20),
        transforms.ColorJitter(contrast=1.0),
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
    """Create PyTorch DataLoaders."""
    
    def seed_worker(worker_id):
        np.random.seed(seed)
        random.seed(seed)
    
    train_dataset = ImageDataset(X_train, y_train, transform=train_transform)
    val_dataset = ImageDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = ImageDataset(X_test, y_test, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            worker_init_fn=seed_worker)

    return train_loader, val_loader, test_loader