"""Bidirectional Gated Cross-Attention mechanism."""

import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention module for multimodal fusion.
    
    This module implements a gated cross-attention mechanism that allows
    one modality to attend to another modality with a gating mechanism
    for controlled information flow.
    """
    
    def __init__(self, query_dim: int, context_dim: int, hidden_dim: int):
        """
        Initialize the Gated Cross-Attention module.
        
        Args:
            query_dim: Dimension of the query modality
            context_dim: Dimension of the context modality
            hidden_dim: Hidden dimension for attention computation
        """
        super(GatedCrossAttention, self).__init__()
        
        # Projection layers for attention
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(context_dim, hidden_dim)
        self.value_proj = nn.Linear(context_dim, hidden_dim)

        # Gating mechanism
        self.gate_fc = nn.Linear(query_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

        # Attention mechanism
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Gated Cross-Attention module.
        
        Args:
            query: Query tensor of shape [B, query_dim]
            context: Context tensor of shape [B, context_dim]
            
        Returns:
            Gated output tensor of shape [B, hidden_dim]
        """
        # Project inputs to hidden space
        Q = self.query_proj(query).unsqueeze(1)     # [B, 1, H]
        K = self.key_proj(context).unsqueeze(1)     # [B, 1, H]
        V = self.value_proj(context).unsqueeze(1)   # [B, 1, H]

        # Compute attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [B, 1, 1]
        attn_weights = self.softmax(attn_scores)       # [B, 1, 1]
        
        # Apply attention to values
        attended = torch.bmm(attn_weights, V).squeeze(1)  # [B, H]

        # Project query into hidden space for fusion
        query_proj = self.query_proj(query)  # [B, H]

        # Compute gate values
        gate_input = torch.cat([query, attended], dim=1)  # [B, query_dim + H]
        gate = self.sigmoid(self.gate_fc(gate_input))     # [B, H]

        # Apply gating mechanism for controlled fusion
        gated_output = gate * query_proj + (1 - gate) * attended  # [B, H]
        
        return gated_output