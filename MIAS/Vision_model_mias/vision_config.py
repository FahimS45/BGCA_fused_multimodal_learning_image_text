"""Configuration for MIAS classification."""

from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Data
    data_dir: str = "/kaggle/input/mias-mammography/all-mias/"
    target_size: tuple = (224, 224)
    
    # Model
    model_name: str = "densenet121"
    num_classes: int = 2
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    step_size: int = 6
    gamma: float = 0.1
    label_smoothing: float = 0.1
    
    # Data splits
    test_size: float = 0.15
    val_size: float = 0.176  # 0.176 of remaining data after test split
    
    # Data augmentation
    scale_factor: int = 2
    augment_minority: bool = True
    
    # Experiment
    seeds: List[int] = None
    n_bootstrap: int = 10000
    ci: int = 95
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 77, 7, 101, 314, 2024, 123, 88, 11, 999]