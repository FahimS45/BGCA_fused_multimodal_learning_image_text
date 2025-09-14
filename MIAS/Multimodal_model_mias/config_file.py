"""Configuration for MIAS Multimodal Classification."""

from dataclasses import dataclass
from typing import List

@dataclass
class MultimodalConfig:
    # Model paths
    densenet_weight_path: str = "/kaggle/input/best-models-with-seeds/pytorch/default/1/DenseNet_best_model.pt"
    deit_weight_path: str = "/kaggle/input/best-models-with-seeds/pytorch/default/1/deit_best_model.bin"
    bert_weight_path: str = "/kaggle/input/best-models-with-seeds/pytorch/default/1/best_BERT_model_state.bin"
    
    # Model configurations
    bert_model_name: str = "Charangan/MedBERT"
    deit_model_name: str = "facebook/deit-base-distilled-patch16-224"
    
    # Data paths
    data_url: str = "/kaggle/input/mias-mammography/all-mias/"
    csv_file_path: str = '/kaggle/input/mias-text-without-co-ordinates/MIAS text.csv'
    
    # Feature dimensions
    densenet_dim: int = 1024
    deit_dim: int = 768
    text_dim: int = 768
    hidden_dim: int = 768
    
    # Network architecture
    fc_input_dim: int = 1024 + 768 + 768  # enhanced_text + enhanced_vision + vision_features[:, :1024]
    fc_hidden_dim: int = 512
    output_dim: int = 2
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_epochs: int = 100
    patience: int = 10
    step_size: int = 6
    gamma: float = 0.1
    label_smoothing: float = 0.1
    
    # Data parameters
    max_length: int = 512
    image_size: tuple = (224, 224)
    scale_factor: int = 2
    
    # Experiment settings
    seeds: List[int] = None
    test_size: float = 0.15
    val_size: float = 0.176
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 77, 7, 101, 314, 2024, 123, 88, 11, 999]