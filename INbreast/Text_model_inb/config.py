"""Configuration for Inbreast text classification."""

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Config:
    # Data
    csv_file: str = "/inbreast/Text-data-annotation.csv"
    text_col: str = "Structured Text"
    label_col: str = "Class"
    label_mapping: Dict[str, int] = None
    
    # Model
    model_name: str = 'emilyalsentzer/Bio_ClinicalBERT'
    max_length: int = 512
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    
    # Training
    batch_size: int = 4
    num_epochs: int = 50
    patience: int = 5
    test_size: float = 0.15
    val_size: float = 0.176
    
    # Experiment
    seeds: List[int] = None
    discriminative_words_n: int = 7
    
    def __post_init__(self):
        if self.label_mapping is None:
            self.label_mapping = {'Non-Malignant': 0, 'Malignant': 1}
        if self.seeds is None:
            self.seeds = [42, 77, 7, 101, 314, 2024, 123, 88, 11, 999]