import torch
import pytest
from decoder_pg import PointerGeneratorDecoder
from model import NL2SQLTransformer

def test_pointer_generator_decoder():
    """Test that the PointerGeneratorDecoder correctly initializes and runs a forward pass."""
    # Initialize decoder with small dimensions for testing
    vocab_size = 100
    d_model = 32
    n_heads = 4
    n_layers = 2
    d_ff = 64
    dropout = 0.0
    pad_token_id = 18
    
    # Create random test inputs
    batch_size = 2
    tgt_seq_len = 7
    src_seq_len = 11
    
    # Create decoder
    dec = PointerGeneratorDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        pad_token_id=pad_token_id
    )
    
    # Create test data
    tgt_ids = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    src_ids = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    memory = torch.randn(batch_size, src_seq_len, d_model)
    
    # Create schema mask (mark first 3 tokens as schema tokens)
    schema_mask = torch.zeros(batch_size, src_seq_len, dtype=torch.bool)
    schema_mask[:, :3] = True
    
    # Run forward pass
    log_probs = dec(
        tgt_ids=tgt_ids,
        src_ids=src_ids,
        memory=memory,
        schema_mask=schema_mask
    )
    
    # Check output shape
    assert log_probs.shape == (batch_size, tgt_seq_len, vocab_size)
    
    # Convert log_probs to probabilities
    probs = log_probs.exp()
    
    # Check that probabilities sum to approximately 1 for each position
    probs_sum = probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)
    
    # Check that the model doesn't generate NaN or inf values
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()

def test_pointer_generator_in_model():
    """Test that the NL2SQLTransformer correctly uses the pointer-generator decoder."""
    # Initialize model with small dimensions for testing
    vocab_size = 100
    d_model = 32
    n_heads = 4
    n_layers = 2
    num_relations = 5
    dropout = 0.0
    pad_token_id = 18
    
    # Create model with pointer-generator
    model = NL2SQLTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        num_relations=num_relations,
        dropout=dropout,
        use_pointer_generator=True,
        pad_token_id=pad_token_id
    )
    
    # Create random test inputs
    batch_size = 2
    seq_len = 11
    
    # Create test data
    encoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    relation_matrix = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long)
    encoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Create schema mask (mark first 3 tokens as schema tokens)
    schema_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    schema_mask[:, :3] = True
    
    # Run forward pass
    outputs = model(
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        encoder_relation_ids=relation_matrix,
        encoder_attention_mask=encoder_attention_mask,
        schema_mask=schema_mask
    )
    
    # Check output
    assert 'log_probs' in outputs, "Model with pointer-generator should return log_probs"
    assert 'encoder_output' in outputs
    
    # Check output shapes
    assert outputs['log_probs'].shape == (batch_size, seq_len, vocab_size)
    assert outputs['encoder_output'].shape == (batch_size, seq_len, d_model)
    
    # Convert log_probs to probabilities and check they sum to 1
    probs = outputs['log_probs'].exp()
    probs_sum = probs.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-3)

if __name__ == "__main__":
    test_pointer_generator_decoder()
    test_pointer_generator_in_model()
    print("All tests passed!")
    
    # Free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 