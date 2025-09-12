"""Pre-trained model loading functions."""

import torch
import torch.nn as nn
from torchvision.models import resnet152
from transformers import BertTokenizer, BertForSequenceClassification, DeiTForImageClassification
from typing import Tuple

def load_resnet_model(weight_path: str) -> nn.Module:
    """
    Load pre-trained ResNet152 model with custom weights.
    
    Args:
        weight_path: Path to the ResNet model weights
        
    Returns:
        Pre-trained ResNet152 model with frozen parameters
    """
    resnet_model = resnet152(pretrained=False)
    num_features = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Identity()  
    
    # Load the state_dict
    state_dict = torch.load(weight_path)
    
    # Remove "fc.weight" and "fc.bias" from the state_dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    
    # Load the pruned state_dict into the model
    resnet_model.load_state_dict(state_dict, strict=False)
    resnet_model.eval()  
    
    # Freeze parameters for feature extraction
    for param in resnet_model.parameters():
        param.requires_grad = False  
    
    return resnet_model

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
    
    # Modify the classifier layer to identity for feature extraction
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

def load_bert_model(weight_path: str, bert_model_name: str = 'dmis-lab/biobert-v1.1') -> Tuple[nn.Module, BertTokenizer]:
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