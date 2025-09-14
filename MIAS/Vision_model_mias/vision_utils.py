"""Utility functions for MIAS classification."""

import torch
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight


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


def calculate_class_weights(labels, device):
    """Calculate class weights for balanced training."""
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return class_weights_tensor