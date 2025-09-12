"""Configuration for Multimodal Inbreast classification."""

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MultimodalConfig:
    # Model paths
    resnet_weight_path: str = "/pytorch/default/resnet_best_model.pth"
    deit_weight_path: str = "/pytorch/default/deit_best_model.pth"
    bert_weight_path: str = "/pytorch/default/best_bert_model_state.bin"
    
    # Model configurations
    bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    deit_model_name: str = "facebook/deit-base-distilled-patch16-224"
    
    # Feature dimensions
    resnet_dim: int = 2048
    deit_dim: int = 768
    text_dim: int = 768
    hidden_dim: int = 512  # Fusion hidden space
    
    # Network architecture
    fc_input_dim: int = 1024  # enhanced_text (512) + enhanced_vision (512)
    fc_hidden_dim: int = 512
    output_dim: int = 2
    num_classes: int = 2
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    patience: int = 5
    
    # Data parameters
    max_length: int = 512  # For text tokenization
    image_size: tuple = (224, 224)
    
    # Experiment settings
    seeds: List[int] = None
    test_size: float = 0.2
    val_size: float = 0.25
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 77, 7, 101, 314, 2024, 123, 88, 11, 999]
        
        # Derived dimensions
        self.vision_dim = self.resnet_dim + self.deit_dim