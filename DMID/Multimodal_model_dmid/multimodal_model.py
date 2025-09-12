"""Complete Multimodal Model with Gated Cross-Attention."""

import torch
import torch.nn as nn
from gated_cross_attention import GatedCrossAttention
from fully_connected_network import FullyConnectedNetwork

class MultiModalModelGatedCrossAttention(nn.Module):
    """
    Multimodal model combining vision and text modalities using 
    bidirectional gated cross-attention mechanism.
    
    The model processes images through both ResNet152 and DeiT models,
    processes text through BERT, and fuses the modalities using
    bidirectional gated cross-attention before final classification.
    """
    
    def __init__(self, resnet_model: nn.Module, deit_model: nn.Module, 
                 text_model: nn.Module, fc_network: FullyConnectedNetwork,
                 resnet_dim: int = 2048, deit_dim: int = 768, 
                 text_dim: int = 768, hidden_dim: int = 512):
        """
        Initialize the Multimodal Model.
        
        Args:
            resnet_model: Pre-trained ResNet152 model
            deit_model: Pre-trained DeiT model
            text_model: Pre-trained BERT model
            fc_network: Fully connected network for final classification
            resnet_dim: ResNet feature dimension
            deit_dim: DeiT feature dimension
            text_dim: Text (BERT) feature dimension
            hidden_dim: Hidden dimension for fusion
        """
        super(MultiModalModelGatedCrossAttention, self).__init__()
        
        # Store pre-trained models
        self.resnet_model = resnet_model
        self.deit_model = deit_model
        self.text_model = text_model
        
        # Store dimensions
        self.resnet_dim = resnet_dim
        self.deit_dim = deit_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Final classification network
        self.fc_network = fc_network
        
        # Combined vision dimension
        self.vision_dim = self.resnet_dim + self.deit_dim

        # Bidirectional gated cross attention
        self.text_to_vision = GatedCrossAttention(
            query_dim=self.text_dim, 
            context_dim=self.vision_dim, 
            hidden_dim=self.hidden_dim
        )
        self.vision_to_text = GatedCrossAttention(
            query_dim=self.vision_dim, 
            context_dim=self.text_dim, 
            hidden_dim=self.hidden_dim
        )

    def forward(self, image_input: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multimodal model.
        
        Args:
            image_input: Input images of shape [B, C, H, W]
            input_ids: Tokenized text input of shape [B, seq_len]
            attention_mask: Attention mask for text input of shape [B, seq_len]
            
        Returns:
            Classification logits of shape [B, num_classes]
        """
        # Extract image features from both vision models
        resnet_features = self.resnet_model(image_input)      # [B, 2048]
        deit_features = self.deit_model(image_input).logits   # [B, 768]
        
        # Concatenate vision features
        vision_features = torch.cat([resnet_features, deit_features], dim=1)  # [B, 2816]

        # Extract text features from [CLS] token of last hidden state
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        text_features = text_outputs.hidden_states[-1][:, 0, :]  # [B, 768]

        # Apply bidirectional gated cross attention
        enhanced_text = self.text_to_vision(text_features, vision_features)    # [B, 512]
        enhanced_vision = self.vision_to_text(vision_features, text_features)  # [B, 512]

        # Concatenate enhanced features for final fusion
        fused_features = torch.cat([enhanced_text, enhanced_vision], dim=1)    # [B, 1024]

        # Final classification
        output = self.fc_network(fused_features)
        return output