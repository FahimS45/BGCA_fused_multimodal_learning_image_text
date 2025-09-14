"""Complete Multimodal Model with Gated Cross-Attention for MIAS dataset."""

import torch
import torch.nn as nn
from gated_cross_attention import GatedCrossAttention
from fully_connected_network import FullyConnectedNetwork

class MultiModalModel(nn.Module):
    """
    Multimodal model combining vision and text modalities using 
    bidirectional gated cross-attention mechanism for MIAS mammography classification.
    
    The model processes images through both DenseNet121 and DeiT models,
    processes text through MedBERT, and fuses the modalities using
    bidirectional gated cross-attention before final classification.
    """
    
    def __init__(self, densenet_model: nn.Module, deit_model: nn.Module, 
                 text_model: nn.Module, fc_network: FullyConnectedNetwork):
        """
        Initialize the Multimodal Model.
        
        Args:
            densenet_model: Pre-trained DenseNet121 model for feature extraction
            deit_model: Pre-trained DeiT model for feature extraction
            text_model: Pre-trained MedBERT model for text feature extraction
            fc_network: Fully connected network for final classification
        """
        super(MultiModalModel, self).__init__()
        
        # Store pre-trained models
        self.densenet_model = densenet_model  
        self.deit_model = deit_model  
        self.text_model = text_model  
        self.fc_network = fc_network  

        # Feature dimensions
        self.image_feature_dim = 1024  # DenseNet output
        self.deit_feature_dim = 768    # DeiT output
        self.text_feature_dim = 768    # MedBERT output
        self.hidden_dim = 768          # Hidden dimension for attention

        # Bidirectional gated cross attention
        self.text_to_vision = GatedCrossAttention(
            query_dim=self.text_feature_dim, 
            context_dim=self.image_feature_dim + self.deit_feature_dim, 
            hidden_dim=self.hidden_dim
        )
        self.vision_to_text = GatedCrossAttention(
            query_dim=self.image_feature_dim + self.deit_feature_dim, 
            context_dim=self.text_feature_dim, 
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
        # Extract features from DenseNet and DeiT
        image_features = self.densenet_model(image_input)    # [B, 1024]
        deit_features = self.deit_model(image_input).logits  # [B, 768]
        
        # Concatenate vision features
        vision_features = torch.cat((image_features, deit_features), dim=1)  # [B, 1792]

        # Extract text features from [CLS] token of last hidden state
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        text_features = text_outputs.hidden_states[-1][:, 0, :]  # [B, 768]

        # Apply bidirectional gated cross attention
        enhanced_text = self.text_to_vision(text_features, vision_features)    # [B, 768]
        enhanced_vision = self.vision_to_text(vision_features, text_features)  # [B, 768]

        # Create final fused features: enhanced_text + enhanced_vision + original image features
        fused_features = torch.cat([
            enhanced_text, 
            enhanced_vision, 
            vision_features[:, :1024]  # First 1024 features from vision (DenseNet)
        ], dim=1)  # [B, 2560]

        # Final classification
        output = self.fc_network(fused_features)
        return output