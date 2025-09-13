"""Configuration for Inbreast vision classification."""

from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Data
    data_dir: str = "/inbreast/Inbreast-images-png"
    target_size: tuple = (224, 224)
    classes: List[str] = None
    
    # Model
    model_name: str = "facebook/deit-base-distilled-patch16-224"
    num_classes: int = 2
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.0001
    weight_decay: float = 1e-3
    
    # Scheduler
    step_size: int = 5
    gamma: float = 0.4
    
    # Data splits
    test_size: float = 0.2
    val_size: float = 0.25  # 0.25 of remaining data after test split
    
    # Experiment
    seeds: List[int] = None
    n_bootstrap: int = 10000
    ci: int = 95
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = ['Non-Malignant_', 'Malignant_']
        if self.seeds is None:
            self.seeds = [42, 77, 7, 101, 314, 2024, 123, 88, 11, 999]