"""Data preprocessing functions for MIAS dataset."""

import os
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from typing import Dict, List, Tuple
from torchvision import transforms

def preprocess_image_2(img: np.ndarray) -> np.ndarray:
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

def apply_bicubic_interpolation(image: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """Enhance an image using bicubic interpolation-based super-resolution."""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    high_res_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    high_res_image = cv2.resize(high_res_image, (224, 224))
    high_res_image = np.expand_dims(high_res_image, axis=-1)
    return high_res_image

def read_images(url: str) -> Dict[str, np.ndarray]:
    """Read and store MIAS mammography images."""
    print("Reading images")
    info = {}
    
    for i in range(322):
        if i < 9:
            image_name = f'mdb00{i + 1}'
        elif i < 99:
            image_name = f'mdb0{i + 1}'
        else:
            image_name = f'mdb{i + 1}'
        
        image_address = os.path.join(url, f"{image_name}.pgm")
        img = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            info[image_name] = img
        else:
            print(f"Warning: Image {image_name} not found.")
    
    print(f"Total images read: {len(info)}")
    return info

def read_labels(url: str) -> Dict[str, int]:
    """Read labels from MIAS info file."""
    print("Reading labels")
    filename = url + 'Info.txt'
    text_all = open(filename).read()
    lines = text_all.split('\n')
    info = {}
    
    for line in lines:
        words = line.split(' ')
        if len(words) > 3:
            if words[3] == 'B':
                info[words[0]] = 0  # Benign
            elif words[3] == 'M':
                info[words[0]] = 1  # Malignant
    
    return info

def balance_dataset(X: np.ndarray, Y: np.ndarray, texts: List[str], 
                   train_transform: transforms.Compose) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Balance the dataset using data augmentation."""
    class_counts = Counter(Y)
    max_count = max(class_counts.values())
    
    # Convert grayscale to RGB
    X_rgb = np.stack([cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB) for img in X])
    
    augmented_images = []
    augmented_labels = []
    augmented_texts = []
    
    for label, count in class_counts.items():
        if count < max_count:
            minority_class_indices = np.where(Y == label)[0]
            images_to_augment = X_rgb[minority_class_indices]
            texts_to_augment = np.array(texts)[minority_class_indices]
            
            while count < max_count:
                for img, text in zip(images_to_augment, texts_to_augment):
                    if count >= max_count:
                        break
                    
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    augmented_img = train_transform(pil_img)
                    
                    augmented_images.append(np.array(augmented_img.permute(1, 2, 0)))
                    augmented_labels.append(label)
                    augmented_texts.append(text)
                    count += 1
    
    if augmented_images:
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        augmented_texts = np.array(augmented_texts)
        
        X_rgb = np.concatenate((X_rgb, augmented_images), axis=0)
        Y = np.concatenate((Y, augmented_labels), axis=0)
        texts = np.concatenate((texts, augmented_texts), axis=0)
    
    return X_rgb, Y, texts

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=60, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=20),
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