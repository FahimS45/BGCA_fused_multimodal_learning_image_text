"""Custom Dataset class for multimodal learning."""

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Any, Dict

class MultiModalDataset(Dataset):
    """
    Custom Dataset for multimodal learning combining images and text.
    
    This dataset handles both image and text data, applying appropriate
    transformations and tokenization for multimodal learning.
    """
    
    def __init__(self, 
                 image_data: List[Image.Image], 
                 text_data: List[str], 
                 labels: List[int], 
                 tokenizer: Any, 
                 transform: Any = None, 
                 max_text_length: int = 512):
        """
        Initialize the MultiModalDataset.
        
        Args:
            image_data: List of PIL Images
            text_data: List of text strings
            labels: List of class labels
            tokenizer: BERT tokenizer for text processing
            transform: Image transformations to apply
            max_text_length: Maximum length for text sequences
        """
        self.image_data = image_data
        self.text_data = text_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing image, text tokens, attention mask, and label
        """
        # Process image data
        image = self.image_data[idx]
        if self.transform:
            image = self.transform(image)
        
        # Process text data
        text = self.text_data[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        # Flatten input IDs and attention masks
        input_ids = encoded_text["input_ids"].flatten()
        attention_mask = encoded_text["attention_mask"].flatten()

        # Get label
        label = self.labels[idx]

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }