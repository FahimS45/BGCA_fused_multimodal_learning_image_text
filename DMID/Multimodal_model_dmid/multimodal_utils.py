"""Utility functions for multimodal model."""

import torch
import numpy as np
import random

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count the total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def print_model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    params = count_parameters(model)
    
    print(f"\n{model_name} Summary:")
    print("-" * 50)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    print(f"Trainable ratio: {params['trainable']/params['total']*100:.2f}%")
    print("-" * 50)

def get_device_info():
    """
    Get information about available computing devices.
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }
    
    return device_info

def print_device_info():
    """Print information about available computing devices."""
    info = get_device_info()
    
    print("\nDevice Information:")
    print("-" * 30)
    print(f"CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"CUDA Device Count: {info['cuda_count']}")
        print(f"Current Device: {info['current_device']}")
        print(f"Device Name: {info['device_name']}")
    else:
        print("Using CPU")
    print("-" * 30)