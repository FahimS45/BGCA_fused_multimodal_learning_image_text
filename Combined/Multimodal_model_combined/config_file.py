"""Configuration for Combined Dataset (DMID+INbreast+MIAS) Multimodal Classification."""

from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class MultimodalConfig:
    # Model paths - these should be updated based on your actual paths
    densenet_weight_path: str = "/kaggle/input/densenet121-best-model-combined/pytorch/default/densenet_best_model.pth"
    deit_weight_path: str = "/kaggle/input/deit-combined/pytorch/default/deit_best_model.pth"
    bert_weight_path: str = "/kaggle/input/bert-best-model-combined/pytorch/default/best_bert_model.bin"
    
    # Data paths
    data_dir: str = "/kaggle/input/combined-dmid-inbreast-mias/Combined_malignant_non-malignant/Combined_malignant_non-malignant"
    csv_path: str = "/kaggle/input/combined-dmid-inbreast-mias/Combined-multimodal-text-data-updated.csv"
    
    # Model configurations
    bert_model_name: str = "bert-base-uncased"
    deit_model_name: str = "facebook/deit-base-distilled-patch16-224"
    
    # Feature dimensions
    densenet_dim: int = 1024  # DenseNet121 classifier input features
    deit_dim: int = 768
    text_dim: int = 768
    hidden_dim: int = 512  # Fusion hidden space
    
    # Network architecture
    fc_input_dim: int = 1024  # enhanced_text (512) + enhanced_vision (512)
    fc_hidden_dim: int = 512
    output_dim: int = 2
    num_classes: int = 2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 100
    patience: int = 10
    step_size: int = 3
    gamma: float = 0.5
    label_smoothing: float = 0.1
    
    # Data parameters
    max_length: int = 512  # For text tokenization
    image_size: tuple = (224, 224)
    
    # Data splitting parameters
    test_size: float = 0.15
    val_size: float = 0.176  # This gives roughly 70-15-15 split
    
    # Experiment settings
    seeds: List[int] = None
    n_bootstrap: int = 10000
    confidence_interval: int = 95
    
    # Class mapping
    label_mapping: Dict[str, int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 77, 7, 101, 314, 2024, 123, 88, 11, 99]
        
        if self.label_mapping is None:
            self.label_mapping = {'Non-Malignant': 0, 'Malignant': 1}
        
        # Derived dimensions
        self.vision_dim = self.densenet_dim + self.deit_dim