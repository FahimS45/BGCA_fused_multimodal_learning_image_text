"""Multimodal Dataset class for MIAS data."""

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional
from torchvision import transforms

class MultiModalDataset(Dataset):
    """
    Dataset class for multimodal mammography data combining images and text.
    
    This dataset handles both image and text data for mammography classification,
    applying appropriate transformations and tokenization.
    """
    
    def __init__(self, images: List, texts: List[str], labels: List[int], 
                 tokenizer, transform: Optional[transforms.Compose] = None, 
                 max_text_length: int = 512):
        """
        Initialize the MultiModalDataset.
        
        Args:
            images: List of image data
            texts: List of text descriptions
            labels: List of labels (0 for benign, 1 for malignant)
            tokenizer: BERT tokenizer for text processing
            transform: Image transformations
            max_text_length: Maximum length for text tokenization
        """
        self.images = images
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_length = max_text_length

    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset."""
        # Process image
        image = self.images[idx]
        
        # Convert numpy array to PIL image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        # Process text
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Get label
        label = self.labels[idx]
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.images)