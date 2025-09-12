"""Factory function to create the complete multimodal model."""

import torch
from config import MultimodalConfig
from model_loaders import load_resnet_model, load_deit_model, load_bert_model
from fully_connected_network import FullyConnectedNetwork
from multimodal_model import MultiModalModelGatedCrossAttention

def create_multimodal_model(config: MultimodalConfig, device: torch.device = None):
    """
    Create and initialize the complete multimodal model.
    
    Args:
        config: Configuration object containing model parameters
        device: Target device for the model
        
    Returns:
        Tuple of (model, tokenizer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained models
    print("Loading pre-trained models...")
    resnet_model = load_resnet_model(config.resnet_weight_path)
    deit_model = load_deit_model(config.deit_weight_path, config.deit_model_name)
    text_model, bert_tokenizer = load_bert_model(config.bert_weight_path, config.bert_model_name)
    
    # Create fully connected network for final classification
    fc_network = FullyConnectedNetwork(
        input_dim=config.fc_input_dim,
        hidden_dim=config.fc_hidden_dim,
        output_dim=config.output_dim
    )
    
    # Create the complete multimodal model
    model = MultiModalModelGatedCrossAttention(
        resnet_model=resnet_model,
        deit_model=deit_model,
        text_model=text_model,
        fc_network=fc_network,
        resnet_dim=config.resnet_dim,
        deit_dim=config.deit_dim,
        text_dim=config.text_dim,
        hidden_dim=config.hidden_dim
    ).to(device)
    
    print(f"Model created and moved to {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, bert_tokenizer, device

def get_model_info(model: MultiModalModelGatedCrossAttention) -> dict:
    """
    Get information about the model architecture.
    
    Args:
        model: The multimodal model
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "vision_dim": model.vision_dim,
        "text_dim": model.text_dim,
        "hidden_dim": model.hidden_dim
    }