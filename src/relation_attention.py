import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class RelationAwareMHA(nn.Module):
    """
    Multi-head attention with a learned bias e_r added to each (i,j) score,
    where e_r is an embedding look-up by relation_ids[b, i, j].
    
    Args:
        d_model (int): Dimension of the model
        n_heads (int): Number of attention heads
        num_relations (int): Number of possible relation types
        dropout (float): Dropout probability
    """
    def __init__(self, d_model: int, n_heads: int, num_relations: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)  # Scale factor for attention scores

        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Relation-type embeddings (one scalar bias per head)
        self.rel_emb = nn.Embedding(num_relations, n_heads)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            
        # Initialize relation embeddings with small values
        nn.init.normal_(self.rel_emb.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, 
                relation_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the relation-aware multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len, seq_len)
            relation_ids (Optional[torch.Tensor]): Relation IDs of shape (batch_size, seq_len, seq_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        B, L, _ = x.size()
        
        # Validate relation_ids if provided
        if relation_ids is not None:
            assert relation_ids.size(1) == L and relation_ids.size(2) == L, \
                f"relation_ids shape must be (batch_size, {L}, {L})"
            assert relation_ids.max() < self.rel_emb.num_embeddings, \
                f"relation_ids values must be < {self.rel_emb.num_embeddings}"

        # Project queries, keys, and values
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B,h,L,d_k)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,h,L,L)

        if relation_ids is not None:
            # Look up bias for each (i,j)
            rel_bias = self.rel_emb(relation_ids).permute(0,3,1,2)  # (B,h,L,L)
            scores = scores + rel_bias

        if attn_mask is not None:
            # Ensure mask is broadcastable
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # (B,1,L,L)
                
            assert attn_mask.size() in [(1,1,L,L), (B,1,L,L)], \
                f"attn_mask shape must be (1,1,{L},{L}) or (batch_size,1,{L},{L})"
                
            scores.masked_fill_(~attn_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,h,L,d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)  # (B,L,d_model)
        return self.out_proj(out)
