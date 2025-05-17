import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TransformerDecoder(nn.Module):
    """Transformer decoder with causal masking for auto-regressive generation."""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, 
                 dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Create decoder layer with custom initialization
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.lm_head.weight)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward(self, tgt_ids: torch.Tensor, encoder_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            tgt_ids: Target token IDs of shape (batch_size, seq_len)
            encoder_out: Encoder output of shape (batch_size, src_len, d_model)
            tgt_mask: Target attention mask of shape (seq_len, seq_len)
            memory_mask: Memory attention mask of shape (seq_len, src_len)
            tgt_key_padding_mask: Target key padding mask of shape (batch_size, seq_len)
            memory_key_padding_mask: Memory key padding mask of shape (batch_size, src_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        seq_len = tgt_ids.size(1)
        
        # Create position indices
        positions = torch.arange(seq_len, device=tgt_ids.device).unsqueeze(0)
        
        # Get embeddings
        tgt_emb = self.token_emb(tgt_ids) + self.pos_emb(positions)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=tgt_ids.device),
                diagonal=1
            ).bool()
            
        # Apply decoder
        x = self.decoder(
            tgt_emb,
            encoder_out,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.lm_head(x)
        return logits