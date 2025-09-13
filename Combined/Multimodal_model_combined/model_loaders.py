"""Pre-trained model loading functions for combined dataset."""

import torch
import torch.nn as nn
from torchvision.models import densenet121
from transformers import BertTokenizer, BertForSequenceClassification, DeiTForImageClassification
from typing import Tuple

def load_densenet_model(weight_path: str) -> nn.Module:
    """
    Load pre-trained DenseNet121 model with custom weights.
    
    Args:
        weight_path: Path to the DenseNet model weights
        
    Returns:
        Pre-trained DenseNet121 model with frozen parameters
    """
    densenet_model = densenet121(pretrained=False)
    num_features = densenet_model.classifier.in_features
    densenet_model.classifier = torch.nn.Identity()  
    
    # Load the state_dict
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))  
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
    
    # Load the pruned state_dict into the model
    densenet_model.load_state_dict(state_dict, strict=False)
    densenet_model.eval() 
    
    for param in densenet_model.parameters():
        param.requires_grad = False  
    
    return densenet_model

def load_deit_model(weight_path: str, model_name: str = "facebook/deit-base-distilled-patch16-224", 
                    num_classes: int = 2) -> nn.Module:
    """
    Load pre-trained DeiT model with custom weights.
    
    Args:
        weight_path: Path to the DeiT model weights
        model_name: Hugging Face model name for DeiT
        num_classes: Number of output classes
        
    Returns:
        Pre-trained DeiT model with frozen parameters
    """
    deit_model = DeiTForImageClassification.from_pretrained(model_name)
    
    # Remove classifier for feature extraction
    deit_model.classifier = nn.Identity()  

    # Load trained weights
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    deit_model.load_state_dict(state_dict, strict=False)

    # Set model to evaluation mode
    deit_model.eval()

    # Freeze parameters for feature extraction
    for param in deit_model.parameters():
        param.requires_grad = False  

    return deit_model

def load_bert_model(weight_path: str, bert_model_name: str = 'bert-base-uncased') -> Tuple[nn.Module, BertTokenizer]:
    """
    Load pre-trained BERT model with custom weights.
    
    Args:
        weight_path: Path to the BERT model weights
        bert_model_name: Hugging Face model name for BERT
        
    Returns:
        Tuple of (BERT model, tokenizer) with frozen parameters
    """
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # Initialize the model with the same configuration used during training
    bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=2)  

    # Load the state_dict into the model
    state_dict = torch.load(weight_path, map_location=torch.device('cpu')) 

    # Load the state dict into the model (ignore mismatched keys if any)
    bert_model.load_state_dict(state_dict, strict=False)

    # Set the model to evaluation mode
    bert_model.eval()

    # Freeze the parameters of the model (for feature extraction)
    for param in bert_model.parameters():
        param.requires_grad = False

    return bert_model, tokenizer