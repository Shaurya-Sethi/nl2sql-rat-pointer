import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from encoder import RelationAwareEncoder
from decoder import TransformerDecoder
from decoder_pg import PointerGeneratorDecoder

class NL2SQLTransformer(nn.Module):
    """
    A transformer-based model for converting natural language to SQL queries.
    
    Args:
        vocab_size (int): Size of the vocabulary
        num_relations (int): Number of possible relation types between tokens
        d_model (int): Dimension of the model
        n_heads (int): Number of attention heads
        n_layers (int): Number of transformer layers
        dropout (float): Dropout probability
        max_len (int): Maximum sequence length
        use_pointer_generator (bool): Whether to use the pointer-generator decoder
        pad_token_id (int): Pad token ID
        cot_start_token_id (Optional[int]): COT start token ID
        sql_end_token_id (Optional[int]): SQL end token ID
    """
    def __init__(self, vocab_size: int, num_relations: int, d_model: int = 768, 
                 n_heads: int = 12, n_layers: int = 12, dropout: float = 0.1, 
                 max_len: int = 2048, use_pointer_generator: bool = False,
                 pad_token_id: int = 18,
                 cot_start_token_id: Optional[int] = None, 
                 sql_end_token_id: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_pointer_generator = use_pointer_generator
        self.pad_token_id = pad_token_id
        self.cot_start_token_id = cot_start_token_id
        self.sql_end_token_id = sql_end_token_id
        
        # Validate required token IDs for generation if not provided (can be made stricter)
        if self.cot_start_token_id is None:
            print("Warning: cot_start_token_id not provided to NL2SQLTransformer. Generation might start with a default token.")
        if self.sql_end_token_id is None:
            print("Warning: sql_end_token_id not provided to NL2SQLTransformer. Generation might not stop correctly.")
        
        # Initialize encoder
        self.encoder = RelationAwareEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            num_relations=num_relations,
            dropout=dropout,
            max_len=max_len
        )
        
        # Initialize decoder based on configuration
        if use_pointer_generator:
            self.decoder = PointerGeneratorDecoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=4*d_model,
                dropout=dropout,
                pad_token_id=pad_token_id,
                max_len=max_len
            )
        else:
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                max_len=max_len
            )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoder_input_ids: torch.Tensor, decoder_input_ids: torch.Tensor,
                encoder_relation_ids: torch.Tensor,
                encoder_attention_mask: Optional[torch.Tensor] = None, # Can be (B,L) padding mask or (B,L,L) attention mask
                decoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_key_padding_mask: Optional[torch.Tensor] = None,
                schema_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            encoder_input_ids: Input token IDs for encoder (batch_size, seq_len)
            decoder_input_ids: Input token IDs for decoder (batch_size, seq_len)
            encoder_relation_ids: Relation IDs between encoder tokens (batch_size, seq_len, seq_len)
            encoder_attention_mask: Optional. If 2D (batch_size, seq_len), it's a padding mask (True for non-padded tokens).
                                      If 3D (batch_size, seq_len, seq_len), it's a self-attention mask.
            decoder_attention_mask: Attention mask for decoder (batch_size, seq_len, seq_len)
            decoder_key_padding_mask: Key padding mask for decoder (batch_size, seq_len)
            schema_mask: Boolean mask indicating schema tokens (batch_size, seq_len)
            
        Returns:
            If using pointer-generator:
                Dictionary containing:
                    - log_probs: Log probabilities (batch_size, seq_len, vocab_size)
                    - encoder_output: Encoder output (batch_size, seq_len, d_model)
            Else:
                Dictionary containing:
                    - logits: Logits for next token prediction (batch_size, seq_len, vocab_size)
                    - encoder_output: Encoder output (batch_size, seq_len, d_model)
        """
        # Validate input shapes
        batch_size, L_enc = encoder_input_ids.shape
        assert encoder_input_ids.size(0) == decoder_input_ids.size(0), \
            "Batch sizes must match between encoder and decoder inputs"
        assert encoder_relation_ids.size(0) == batch_size, \
            "Batch size must match for relation IDs"
        assert encoder_relation_ids.size(1) == encoder_relation_ids.size(2) == L_enc, \
            "Relation IDs shape must match encoder sequence length"

        # Ensure proper dtype for tensors, especially important for mixed precision training
        # Integer tensors should remain as long, boolean masks should be boolean
        encoder_input_ids = encoder_input_ids.long()
        decoder_input_ids = decoder_input_ids.long()
        encoder_relation_ids = encoder_relation_ids.long()
        
        # Convert masks to boolean if not already
        if encoder_attention_mask is not None:
            if encoder_attention_mask.dtype != torch.bool:
                encoder_attention_mask = encoder_attention_mask.bool()
        
        if decoder_attention_mask is not None:
            if decoder_attention_mask.dtype != torch.bool:
                decoder_attention_mask = decoder_attention_mask.bool()
        
        if decoder_key_padding_mask is not None:
            if decoder_key_padding_mask.dtype != torch.bool:
                decoder_key_padding_mask = decoder_key_padding_mask.bool()
        
        if schema_mask is not None:
            if schema_mask.dtype != torch.bool:
                schema_mask = schema_mask.bool()

        # Process encoder_attention_mask for encoder self-attention
        # The encoder's self-attention layers expect a (B, L_enc, L_enc) mask.
        encoder_self_attn_mask_for_encoder_layers = None
        if encoder_attention_mask is not None:
            if encoder_attention_mask.dim() == 2:  # (B, L_enc) padding mask (True for non-padded)
                # Expand padding mask to be (B, L_enc, L_enc) for self-attention.
                # This allows a query token i to attend to key token j if key token j is not padded.
                encoder_self_attn_mask_for_encoder_layers = encoder_attention_mask.unsqueeze(1).expand(-1, L_enc, -1)
            elif encoder_attention_mask.dim() == 3:  # Already a (B, L_enc, L_enc) self-attention mask
                encoder_self_attn_mask_for_encoder_layers = encoder_attention_mask
            else:
                raise ValueError(
                    f"encoder_attention_mask has unexpected dimensions: {encoder_attention_mask.shape}"
                )
            
        # Encode input
        encoder_output = self.encoder(
            input_ids=encoder_input_ids,
            relation_ids=encoder_relation_ids,
            attention_mask=encoder_self_attn_mask_for_encoder_layers # Pass the (B, L_enc, L_enc) mask
        )
        
        # Create key padding mask for decoder if not provided (True for padded tokens)
        if decoder_key_padding_mask is None:
            decoder_key_padding_mask = (decoder_input_ids == self.pad_token_id)
        
        # Determine memory_key_padding_mask for decoder's cross-attention (True for padded encoder tokens)
        # This uses the original encoder_input_ids to identify padding.
        actual_memory_key_padding_mask = (encoder_input_ids == self.pad_token_id)

        # Decode with appropriate decoder
        if self.use_pointer_generator:
            # Ensure schema mask is provided for pointer-generator
            assert schema_mask is not None, "Schema mask is required for pointer-generator decoder"
            
            # Forward through pointer-generator decoder
            log_probs = self.decoder(
                tgt_ids=decoder_input_ids,
                src_ids=encoder_input_ids,
                memory=encoder_output,
                schema_mask=schema_mask,
                tgt_mask=decoder_attention_mask, # This is the causal mask for decoder self-attention
                tgt_key_padding_mask=decoder_key_padding_mask, # Padding for target sequence
                memory_key_padding_mask=actual_memory_key_padding_mask # Padding for encoder sequence in cross-attention
            )
            
            return {
                'log_probs': log_probs,
                'encoder_output': encoder_output
            }
        else:
            # Forward through standard decoder
            logits = self.decoder(
                tgt_ids=decoder_input_ids,
                encoder_out=encoder_output,
                tgt_mask=decoder_attention_mask, # This is the causal mask for decoder self-attention
                tgt_key_padding_mask=decoder_key_padding_mask, # Padding for target sequence
                memory_key_padding_mask=actual_memory_key_padding_mask # Padding for encoder sequence in cross-attention
            )
            
            return {
                'logits': logits,
                'encoder_output': encoder_output
            }
        
    def generate(self, encoder_input_ids: torch.Tensor, encoder_relation_ids: torch.Tensor,
                max_length: int = 1024, # Default to 1024 for full COT+SQL
                encoder_attention_mask: Optional[torch.Tensor] = None,
                schema_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate SQL query from natural language input using greedy decoding.
        
        Args:
            encoder_input_ids: Input token IDs for encoder (batch_size, seq_len)
            encoder_relation_ids: Relation IDs between encoder tokens (batch_size, seq_len, seq_len)
            max_length: Maximum length of generated sequence
            encoder_attention_mask: Optional. If 2D (batch_size, seq_len), it's a padding mask (True for non-padded tokens).
                                      If 3D (batch_size, seq_len, seq_len), it's a self-attention mask.
            schema_mask: Boolean mask indicating schema tokens (batch_size, seq_len)
            
        Returns:
            Generated token IDs (batch_size, max_length)
        """
        # Validate input
        batch_size, L_enc = encoder_input_ids.shape
        device = encoder_input_ids.device
        
        # Process encoder_attention_mask for encoder self-attention
        encoder_self_attn_mask_for_encoder_layers = None
        if encoder_attention_mask is not None:
            if encoder_attention_mask.dim() == 2:  # (B, L_enc) padding mask (True for non-padded)
                encoder_self_attn_mask_for_encoder_layers = encoder_attention_mask.unsqueeze(1).expand(-1, L_enc, -1)
            elif encoder_attention_mask.dim() == 3:  # Already a (B, L_enc, L_enc) self-attention mask
                encoder_self_attn_mask_for_encoder_layers = encoder_attention_mask
            else:
                raise ValueError(
                    f"encoder_attention_mask has unexpected dimensions: {encoder_attention_mask.shape}"
                )
        
        # Encode input once
        encoder_output = self.encoder(
            input_ids=encoder_input_ids,
            relation_ids=encoder_relation_ids,
            attention_mask=encoder_self_attn_mask_for_encoder_layers
        )
        
        # Initialize decoder input with COT_START token ID
        if self.cot_start_token_id is None:
            print("Warning: cot_start_token_id is None in model.generate. Using pad_token_id + 1 as fallback BOS.")
            start_token_id = self.pad_token_id + 1 
        else:
            start_token_id = self.cot_start_token_id
            
        decoder_input = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        # Memory key padding mask for cross attention in decoder (True for padded encoder tokens)
        memory_key_padding_mask = (encoder_input_ids == self.pad_token_id)
        
        # Generation loop
        for _ in range(max_length - 1): # -1 because we already have the start token
            # Create causal mask for decoder self-attention
            seq_len = decoder_input.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            
            # Decoder key padding mask for its self-attention (True for padded decoder tokens)
            current_tgt_key_padding_mask = (decoder_input == self.pad_token_id)

            # Forward pass through decoder
            if self.use_pointer_generator:
                # Pointer-generator decoder expects src_ids and schema_mask
                if schema_mask is None:
                    # If PG is used, schema_mask should ideally be provided.
                    # Create a dummy all-False mask if not, but this might affect PG performance.
                    print("Warning: schema_mask is None in model.generate for pointer-generator. Using all-False mask.")
                    current_schema_mask = torch.zeros_like(encoder_input_ids, dtype=torch.bool, device=device)
                else:
                    current_schema_mask = schema_mask

                outputs = self.decoder(
                    tgt_ids=decoder_input,
                    src_ids=encoder_input_ids, # For pointer mechanism
                    memory=encoder_output,
                    schema_mask=current_schema_mask, # For pointer mechanism
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=current_tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                # PG decoder returns log_probs
                next_token_logits = outputs[:, -1:, :] # Get the logits for the last token
                next_token = next_token_logits.argmax(dim=-1) # Greedy decoding
            else:
                # Standard decoder
                outputs = self.decoder(
                    tgt_ids=decoder_input,
                    encoder_out=encoder_output,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=current_tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                # Standard decoder returns logits
                next_token_logits = outputs[:, -1:, :] # Get the logits for the last token
                next_token = next_token_logits.argmax(dim=-1) # Greedy decoding
            
            # Append next token to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Stop if SQL_END token is generated (if defined)
            if self.sql_end_token_id is not None and (next_token == self.sql_end_token_id).all():
                break
        
        return decoder_input
