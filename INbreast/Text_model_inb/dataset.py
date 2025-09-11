"""Dataset and data loading."""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def load_data(csv_file: str, text_col: str, label_col: str, label_mapping: dict) -> Tuple[List[str], List[int]]:
    """Load and prepare data."""
    df = pd.read_csv(csv_file)
    texts = df[text_col].tolist()
    labels = df[label_col].map(label_mapping).values
    return texts, labels

def split_data(texts: List[str], labels: List[int], test_size: float, val_size: float, seed: int):
    """Split data into train/val/test."""
    texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_train_val, labels_train_val, test_size=val_size, 
        stratify=labels_train_val, random_state=seed
    )
    
    return texts_train, texts_val, texts_test, labels_train, labels_val, labels_test

def create_dataloaders(texts_train, texts_val, texts_test, labels_train, labels_val, labels_test,
                      tokenizer, batch_size: int, max_length: int = 512):
    """Create PyTorch DataLoaders."""
    train_dataset = TextDataset(texts_train, labels_train, tokenizer, max_length)
    val_dataset = TextDataset(texts_val, labels_val, tokenizer, max_length)
    test_dataset = TextDataset(texts_test, labels_test, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader