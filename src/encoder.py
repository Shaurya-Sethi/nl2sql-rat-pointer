import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from relation_attention import RelationAwareMHA

class RelationAwareEncoderLayer(nn.Module):
    """A single layer of the relation-aware encoder."""
    
    def __init__(self, d_model: int, n_heads: int, num_relations: int, dropout: float):
        super().__init__()
        self.self_attn = RelationAwareMHA(d_model, n_heads, num_relations, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, 
                relation_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            relation_ids: Relation IDs of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(
            self.self_attn(x, attn_mask=attn_mask, relation_ids=relation_ids)
        )
        
        # Feed-forward with residual connection and layer norm
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        
        return x

class RelationAwareEncoder(nn.Module):
    """Relation-aware encoder with multiple layers."""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 num_relations: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            RelationAwareEncoderLayer(d_model, n_heads, num_relations, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, relation_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            relation_ids: Relation IDs of shape (batch_size, seq_len, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, L = input_ids.shape
        
        # Create position indices
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        
        # Get embeddings
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(B, L, L, device=input_ids.device, dtype=torch.bool)
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask, relation_ids=relation_ids)
            
        return self.norm(x)  # Final layer norm
