"""Fully Connected Network for final classification."""

import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    """
    Fully Connected Network for final multimodal classification.
    
    This network takes the fused multimodal features and produces
    the final classification output through a series of fully
    connected layers with batch normalization and dropout.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the Fully Connected Network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
        """
        super(FullyConnectedNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Second layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            # Output layer
            nn.Linear(hidden_dim // 2, output_dim)          
        )

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape [B, input_dim]
            
        Returns:
            Output logits of shape [B, output_dim]
        """
        return self.fc(x)