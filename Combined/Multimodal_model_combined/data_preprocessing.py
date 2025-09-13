"""Data preprocessing utilities for combined multimodal dataset."""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms
import torch
from typing import List, Tuple, Dict

def load_and_preprocess_images(data_dir: str, target_size: Tuple[int, int] = (224, 224)) -> List[Image.Image]:
    """
    Load and preprocess images from directory.
    
    Args:
        data_dir: Directory containing images
        target_size: Target image size (height, width)
        
    Returns:
        List of preprocessed PIL Images
    """
    X = []
    
    # Sort filenames to maintain the default order
    for img_name in sorted(os.listdir(data_dir)):
        img_path = os.path.join(data_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')  
                img = img.resize(target_size)  
                X.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return X

def create_transforms():
    """
    Create image transforms for training and validation/test.
    
    Returns:
        Tuple of (train_transform, val_test_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute balanced class weights for handling class imbalance.
    
    Args:
        labels: Array of class labels
        device: Target device for the tensor
        
    Returns:
        Tensor of class weights on the specified device
    """
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=torch.unique(torch.tensor(labels)).numpy(),
        y=labels
    )

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"Class Weights on Device: {class_weights_tensor}")
    return class_weights_tensor

def preprocess_multimodal_data(df: pd.DataFrame, X: List, seed: int = 42) -> Tuple:
    """
    Preprocess multimodal data with proper group-aware splitting for INbreast dataset.
    
    Args:
        df: DataFrame containing text data and metadata
        X: List of images
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of split data: (texts_train, labels_train, images_train,
                            texts_val, labels_val, images_val,
                            texts_test, labels_test, images_test)
    """
    # Step 1: Split dataframe by source
    df_inbreast = df[df['source'] == 'INbreast'].copy()
    df_others = df[df['source'] != 'INbreast'].copy()

    # Step 2: Extract report_id for GroupShuffleSplit
    def extract_report_id(filename):
        parts = filename.split('_')
        return parts[1] if len(parts) > 2 else None

    df_inbreast['report_id'] = df_inbreast['Image Name'].apply(extract_report_id)

    # Step 3: Prepare INbreast variables
    inb_texts = df_inbreast['Generated Sentence'].values  # Updated column name
    inb_labels = df_inbreast['Class'].values
    inb_images = X[:308]  # First 308 images are INbreast
    inb_groups = df_inbreast['report_id'].values

    # Step 4: INbreast GroupShuffleSplit (Train+Val vs Test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(gss.split(inb_texts, inb_labels, groups=inb_groups))

    inb_texts_train_val, inb_texts_test = inb_texts[train_val_idx], inb_texts[test_idx]
    inb_images_train_val = [inb_images[i] for i in train_val_idx]
    inb_images_test = [inb_images[i] for i in test_idx]
    inb_labels_train_val, inb_labels_test = inb_labels[train_val_idx], inb_labels[test_idx]
    group_ids_train_val = inb_groups[train_val_idx]

    # Step 5: INbreast Train vs Val GroupShuffleSplit
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=seed)
    train_idx, val_idx = next(gss2.split(inb_texts_train_val, inb_labels_train_val, groups=group_ids_train_val))

    inb_texts_train, inb_texts_val = inb_texts_train_val[train_idx], inb_texts_train_val[val_idx]
    inb_images_train = [inb_images_train_val[i] for i in train_idx]
    inb_images_val = [inb_images_train_val[i] for i in val_idx]
    inb_labels_train, inb_labels_val = inb_labels_train_val[train_idx], inb_labels_train_val[val_idx]

    # Step 6: DMID & MIAS Train vs Val split
    texts_others = df_others['Generated Sentence'].values  # Updated column name
    labels_others = df_others['Class'].values
    images_others = X[308:]  # Remaining images are DMID + MIAS

    texts_others_train_val, texts_others_test, labels_others_train_val, labels_others_test, images_others_train_val, images_others_test = train_test_split(
        texts_others, labels_others, images_others, test_size=0.15, stratify=labels_others, random_state=seed
    )

    texts_others_train, texts_others_val, labels_others_train, labels_others_val, images_others_train, images_others_val = train_test_split(
        texts_others_train_val, labels_others_train_val, images_others_train_val, test_size=0.176, stratify=labels_others_train_val, random_state=seed
    )

    # Step 8: Final concatenation
    texts_train = np.concatenate([inb_texts_train, texts_others_train])
    labels_train = np.concatenate([inb_labels_train, labels_others_train])
    images_train = np.concatenate([inb_images_train, images_others_train])

    texts_val = np.concatenate([inb_texts_val, texts_others_val])
    labels_val = np.concatenate([inb_labels_val, labels_others_val])
    images_val = np.concatenate([inb_images_val, images_others_val])

    texts_test = np.concatenate([inb_texts_test, texts_others_test])
    labels_test = np.concatenate([inb_labels_test, labels_others_test])
    images_test = np.concatenate([inb_images_test, images_others_test])

    return texts_train, labels_train, images_train, texts_val, labels_val, images_val, texts_test, labels_test, images_test

def load_data(config) -> Tuple[List, np.ndarray, List]:
    """
    Load and prepare all data according to configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (images, labels, texts)
    """
    # Load images
    images = load_and_preprocess_images(config.data_dir)
    
    # Load text data
    df = pd.read_csv(config.csv_path)
    
    # Process labels and texts
    labels = df['Class'].map(config.label_mapping).values
    texts = df['Generated Sentence'].tolist()  # Updated column name
    
    print(f"Loaded {len(images)} images, {len(texts)} texts")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return images, labels, texts, df