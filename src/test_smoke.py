import os
import logging
import torch
from pathlib import Path
from model import NL2SQLTransformer
# No Tokenizer needed for this basic smoke test with dummy data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DummyDataset(torch.utils.data.Dataset):
    """Creates a few dummy samples of token IDs for smoke testing."""
    def __init__(self, num_samples=10, max_len=64, vocab_size=100, pad_token_id=18):
        self.samples = []
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        for _ in range(num_samples):
            seq_len = torch.randint(10, max_len + 1, (1,)).item()
            ids = torch.randint(0, vocab_size, (seq_len,), dtype=torch.long)
            self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        seq_len = input_ids.size(0)
        
        # Dummy attention mask (True for non-padded, assuming no padding in these short sequences for simplicity)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Dummy relation matrix (all zeros)
        relation_matrix = torch.zeros((seq_len, seq_len), dtype=torch.long)
        
        # Dummy schema mask (e.g., first few tokens or all False)
        schema_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if seq_len > 3:
            schema_mask[:3] = True # Mark first 3 as dummy schema tokens
            
        return {
            'encoder_input_ids': input_ids,
            'decoder_input_ids': input_ids.clone(), # Use same for LM-style smoke test
            'encoder_attention_mask': attention_mask,
            'encoder_relation_ids': relation_matrix,
            'schema_mask': schema_mask
        }

def collate_fn_smoke(batch, pad_id=18):
    """Pads and collates a batch of dummy data for the smoke test."""
    # Determine max length in batch for encoder and decoder inputs
    max_len_enc = max(len(item['encoder_input_ids']) for item in batch)
    max_len_dec = max(len(item['decoder_input_ids']) for item in batch)
    max_len = max(max_len_enc, max_len_dec)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    encoder_input_padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    decoder_input_padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    encoder_attention_mask_padded = torch.zeros(batch_size, max_len, dtype=torch.bool) # False for padded positions
    encoder_relation_ids_padded = torch.zeros(batch_size, max_len, max_len, dtype=torch.long)
    schema_mask_padded = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        enc_len = item['encoder_input_ids'].size(0)
        dec_len = item['decoder_input_ids'].size(0)
        
        encoder_input_padded[i, :enc_len] = item['encoder_input_ids']
        decoder_input_padded[i, :dec_len] = item['decoder_input_ids']
        encoder_attention_mask_padded[i, :enc_len] = item['encoder_attention_mask'] # True for non-padded
        
        # Pad relation_ids and schema_mask (associated with encoder_input)
        encoder_relation_ids_padded[i, :enc_len, :enc_len] = item['encoder_relation_ids']
        schema_mask_padded[i, :enc_len] = item['schema_mask']
        
    return {
        'encoder_input_ids': encoder_input_padded,
        'decoder_input_ids': decoder_input_padded,
        'encoder_attention_mask': encoder_attention_mask_padded,
        'encoder_relation_ids': encoder_relation_ids_padded,
        'schema_mask': schema_mask_padded
    }

def test_smoke_setup():
    logger.info("Starting smoke test setup...")
    try:
        # Minimal model config for smoke test
        vocab_size = 100  # Small vocab for dummy data
        d_model = 32      # Small dimensions for speed
        n_heads = 2
        n_layers = 1
        num_relations = 5 # Consistent with RelationMatrixBuilder expectation
        dropout = 0.1
        max_len_model = 128 # Max length the model supports
        pad_token_id = 18
        use_pg = True     # Test with Pointer-Generator enabled

        model = NL2SQLTransformer(
            vocab_size=vocab_size,
            num_relations=num_relations,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len_model,
            use_pointer_generator=use_pg,
            pad_token_id=pad_token_id
        )
        logger.info(f"NL2SQLTransformer initialized successfully (Pointer-Generator: {use_pg})")

        # Use the DummyDataset for smoke test
        # Dataset max_len should be <= model max_len
        dataset_max_len = 64 
        dataset = DummyDataset(num_samples=4, max_len=dataset_max_len, vocab_size=vocab_size, pad_token_id=pad_token_id)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=lambda b: collate_fn_smoke(b, pad_id=pad_token_id))
        
        if not dataset:
            logger.error("DummyDataset is empty!")
            return False
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            logger.error("DataLoader is empty, cannot get a batch.")
            return False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logger.info(f"Batch loaded to device: {device}")

        logger.info("Running forward pass for smoke test...")
        
        # Decoder input for teacher forcing style, excluding the last token
        decoder_input_for_forward = batch['decoder_input_ids'][:, :-1]
        
        outputs = model(
            encoder_input_ids=batch['encoder_input_ids'],
            decoder_input_ids=decoder_input_for_forward,
            encoder_relation_ids=batch['encoder_relation_ids'],
            encoder_attention_mask=batch['encoder_attention_mask'], # This should be (B,L) or (B,L,L)
            schema_mask=batch['schema_mask']
        )
        logger.info("Forward pass successful!")
        
        output_key = 'log_probs' if use_pg else 'logits'
        assert output_key in outputs, f"Expected output key '{output_key}' not found."
        
        # Expected shape: (batch_size, decoder_input_seq_len, vocab_size)
        # decoder_input_seq_len is batch['decoder_input_ids'][:, :-1].size(1)
        expected_dec_seq_len = decoder_input_for_forward.size(1)

        logger.info(f"Output tensor ({output_key}) shape: {outputs[output_key].shape}")
        assert outputs[output_key].shape == (batch['encoder_input_ids'].shape[0], expected_dec_seq_len, vocab_size), \
            f"Shape mismatch for {output_key}. Got {outputs[output_key].shape}, expected {(batch['encoder_input_ids'].shape[0], expected_dec_seq_len, vocab_size)}"
        
        logger.info(f"Encoder output shape: {outputs['encoder_output'].shape}")
        assert outputs['encoder_output'].shape == (batch['encoder_input_ids'].shape[0], batch['encoder_input_ids'].shape[1], d_model)
        
        logger.info("Smoke test PASSED!")
        
        # Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
    except Exception as e:
        logger.error(f"Smoke test FAILED: {str(e)}", exc_info=True)
        
        # Free up GPU memory even after error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return False

if __name__ == '__main__':
    test_smoke_setup() 