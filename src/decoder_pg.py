"""Pointer-Generator aware Transformer decoder stack.

This implementation allows copying tokens directly from the encoder input
based on attention weights, especially useful for schema tokens like table
and column names in SQL generation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Helper: returns attn + hidden
# -----------------------------
class _PGDecoderLayer(nn.Module):
    """Single Transformer decoder layer that **returns** cross-attention
    weights (`attn_w`) in addition to the hidden states `h`.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,            # (B, T, d_model)
        memory: torch.Tensor,         # (B, S, d_model)
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        _tgt, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.norm1(tgt + self.dropout(_tgt))

        # Cross-attention  (return weights for pointer-generator)
        _tgt, attn_w = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # per-head weights (B, nH, T, S)
        )
        tgt = self.norm2(tgt + self.dropout(_tgt))

        # Feed-forward
        _ff = self.lin2(self.dropout(self.activation(self.lin1(tgt))))
        tgt = self.norm3(tgt + self.dropout(_ff))
        # average heads → (B,T,S)
        attn_w = attn_w.mean(dim=1)
        return tgt, attn_w


class PointerGeneratorDecoder(nn.Module):
    """N-layer Transformer decoder with copy mechanism focused on schema tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            _PGDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.lm_head.weight)

        # Pointer-generator gate
        self.p_gen = nn.Linear(d_model * 3, 1)
        nn.init.xavier_uniform_(self.p_gen.weight)
        nn.init.zeros_(self.p_gen.bias)

    # ---------------------------------------------------------
    def forward(
        self,
        tgt_ids: torch.Tensor,             # (B,T)
        src_ids: torch.Tensor,             # (B,S)
        memory: torch.Tensor,              # encoder hidden (B,S,d)
        schema_mask: torch.Tensor,         # (B,S) bool – True where token is table/col name
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns **log-probs** over extended vocab (`P_final`)."""

        B, T = tgt_ids.shape
        device = tgt_ids.device

        # ----- Embedding + pos enc -----
        # Create position indices
        positions = torch.arange(T, device=device).unsqueeze(0)
        
        # Get embeddings
        x = self.token_emb(tgt_ids) + self.pos_emb(positions)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = torch.triu(
                torch.ones(T, T, device=device),
                diagonal=1
            ).bool()

        # Pass through decoder layers
        attn_w_last = None  # keep weights from final layer
        for layer in self.layers:
            x, attn_w = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            attn_w_last = attn_w  # Save the attention weights from the final layer

        assert attn_w_last is not None, "attention weights missing"
        # x: (B,T,d) ; attn_w_last: (B,T,S)

        # ----- P_vocab (standard vocabulary distribution) -----
        P_vocab = F.softmax(self.lm_head(x), dim=-1)                 # (B,T,V)

        # ----- P_copy (schema-only distribution) -----
        # Convert schema mask to float and apply to attention weights
        schema_mask_f = schema_mask.to(dtype=x.dtype)                # (B,S)
        # Mask out non-schema tokens in attention weights
        P_copy_src = attn_w_last * schema_mask_f.unsqueeze(1)        # (B,T,S)
        # Renormalize over schema positions
        denom = P_copy_src.sum(-1, keepdim=True) + 1e-8
        P_copy_src = P_copy_src / denom                              # renorm over schema positions

        # Scatter attention weights to vocabulary space using source token IDs
        V = P_vocab.size(-1)
        P_copy_vocab = torch.zeros_like(P_vocab)                     # (B,T,V)
        expand_ids = src_ids.unsqueeze(1).expand(-1, T, -1)          # (B,T,S)
        P_copy_vocab.scatter_add_(2, expand_ids, P_copy_src)         # add prob mass

        # ----- Generate vs Copy gate -----
        # Compute context vector by weighting memory with attention
        c_t = torch.bmm(P_copy_src, memory)                          # (B,T,d)
        # Gate input combines decoder state, context vector, and input embedding
        gate_inp = torch.cat([x, c_t, self.token_emb(tgt_ids)], dim=-1)  # (B,T,3d)
        p_gen = torch.sigmoid(self.p_gen(gate_inp))                  # (B,T,1)

        # ----- Final distribution combining generation and copying -----
        P_final = p_gen * P_vocab + (1.0 - p_gen) * P_copy_vocab
        log_P_final = torch.log(P_final + 1e-9)
        return log_P_final 