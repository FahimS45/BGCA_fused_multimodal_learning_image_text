"""Utility functions for vision model."""

import torch
import numpy as np
import random
from collections import Counter

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_class_weights(train_loader, device):
    """Calculate class weights for balanced training."""
    # Extract labels from train_loader
    labels = []
    for _, batch_labels in train_loader:
        labels.extend(batch_labels.tolist())
    
    # Count class occurrences
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    
    # Calculate weights: inverse frequency
    class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
    
    # Convert to PyTorch tensor and move to device
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    return class_weights