import os
import logging
import torch
from pathlib import Path
import json
from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from relation_matrix import RelationMatrixBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_directory_structure():
    """Verify all required directories exist."""
    required_dirs = [
        'datasets/paired_nl_sql/splits',
        'datasets/raw_sql/pretraining corpus/splits',
        'models',
        'outputs'
    ]
    
    project_root = Path(__file__).resolve().parent.parent
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(str(full_path)) # Log full path for clarity
        else:
            logger.info(f"✓ Directory exists: {full_path}")
            
    return missing_dirs

def verify_data_files():
    """Verify all required data files exist and are readable."""
    required_files = [
        'datasets/paired_nl_sql/splits/tokenized_sft_filtered_train.txt',
        'datasets/paired_nl_sql/splits/tokenized_sft_filtered_val.txt',
        'datasets/raw_sql/pretraining corpus/splits/wrapped_tokenized_corpus_train.txt',
        'datasets/raw_sql/pretraining corpus/splits/wrapped_tokenized_corpus_val.txt'
    ]
    
    project_root = Path(__file__).resolve().parent.parent
    missing_files = []
    unreadable_files = []
    
    for file_path_str in required_files:
        full_path = project_root / file_path_str
        if not full_path.exists():
            missing_files.append(str(full_path))
        else:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line.strip(): # Check if first line is empty after stripping
                        # Allow empty files, but log a warning if it might be an issue
                        logger.warning(f"⁇ File exists but first line is empty (or whitespace only): {full_path}")
                    logger.info(f"✓ File exists and readable: {full_path}")
            except Exception as e:
                unreadable_files.append(f"{full_path} (Error: {str(e)})")
                
    return missing_files, unreadable_files

def verify_model_files():
    """Verify model files exist."""
    required_files = [
        'models/nl2sql_tok.model'
    ]
    
    project_root = Path(__file__).resolve().parent.parent
    missing_files = []
    
    for file_path_str in required_files:
        full_path = project_root / file_path_str
        if not full_path.exists():
            missing_files.append(str(full_path))
        else:
            logger.info(f"✓ Model file exists: {full_path}")
            
    return missing_files

def verify_gpu():
    """Verify GPU availability and memory."""
    if not torch.cuda.is_available():
        logger.info("ℹ Running in CPU mode (no GPU available)")
        logger.info("  This is expected for local testing without a dedicated GPU.")
        logger.info("  GPU will be required for efficient cloud training.")
        return True
        
    gpu_count = torch.cuda.device_count()
    logger.info(f"✓ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
    return True

def verify_model_components():
    """Verify model components can be initialized and a forward pass runs."""
    logger.info("Verifying model component initialization...")
    try:
        # Minimal config for testing model instantiation
        vocab_size = 100  # Small vocab for dummy test
        d_model = 32
        n_heads = 2
        n_layers = 1
        num_relations = 5
        dropout = 0.1
        max_len_model = 64
        pad_token_id = 18 # Standard pad token ID
        use_pg_for_test = False # Test non-PG path for basic component check
        
        model = NL2SQLTransformer(
            vocab_size=vocab_size,
            num_relations=num_relations,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len_model,
            use_pointer_generator=use_pg_for_test,
            pad_token_id=pad_token_id
        )
        logger.info(f"✓ NL2SQLTransformer initialized (Pointer-Generator: {use_pg_for_test})")
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len_test = 32 # Must be <= max_len_model
        
        # Dummy inputs
        encoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len_test), dtype=torch.long)
        # For non-PG decoder, decoder_input_ids are usually shifted targets
        decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len_test -1), dtype=torch.long) # Typical for teacher forcing
        encoder_relation_ids = torch.zeros(batch_size, seq_len_test, seq_len_test, dtype=torch.long)
        # Dummy encoder_attention_mask (True for non-padded, all True for this test)
        encoder_attention_mask = torch.ones(batch_size, seq_len_test, dtype=torch.bool)
        schema_mask_dummy = None # PG is false, so can be None
        if use_pg_for_test:
            schema_mask_dummy = torch.zeros(batch_size, seq_len_test, dtype=torch.bool) # Create if testing PG

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        encoder_relation_ids = encoder_relation_ids.to(device)
        encoder_attention_mask = encoder_attention_mask.to(device)
        if schema_mask_dummy is not None:
            schema_mask_dummy = schema_mask_dummy.to(device)

        logger.info("Attempting model forward pass...")
        outputs = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids, # Corrected length for decoder input
            encoder_relation_ids=encoder_relation_ids,
            encoder_attention_mask=encoder_attention_mask,
            schema_mask=schema_mask_dummy
        )
        
        output_key = 'logits' # Since use_pg_for_test is False
        assert output_key in outputs, f"Expected output key '{output_key}' not found."
        logger.info(f"✓ Model forward pass successful. Output key '{output_key}' found.")
        logger.info(f"  {output_key} shape: {outputs[output_key].shape}")
        # Expected output shape: (batch_size, decoder_seq_len, vocab_size)
        # decoder_seq_len is decoder_input_ids.size(1) which is seq_len_test - 1
        assert outputs[output_key].shape == (batch_size, seq_len_test - 1, vocab_size)

        logger.info(f"  Encoder output shape: {outputs['encoder_output'].shape}")
        assert outputs['encoder_output'].shape == (batch_size, seq_len_test, d_model)
        
        logger.info("✓ Model components verified successfully.")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error verifying model components: {str(e)}", exc_info=True)
        return False

def main():
    logger.info("Starting comprehensive setup verification...")
    all_checks_passed = True
    
    # 1. Check directory structure
    logger.info("\n1. Verifying directory structure...")
    missing_dirs = verify_directory_structure()
    if missing_dirs:
        all_checks_passed = False
        logger.error("✗ Missing directories:")
        for dir_path in missing_dirs:
            logger.error(f"  - {dir_path}")
    else:
        logger.info("✓ All required directories exist")
        
    # 2. Check data files
    logger.info("\n2. Verifying data files...")
    missing_files, unreadable_files = verify_data_files()
    if missing_files:
        all_checks_passed = False
        logger.error("✗ Missing data files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
    if unreadable_files:
        all_checks_passed = False
        logger.error("✗ Unreadable data files:")
        for file_path in unreadable_files:
            logger.error(f"  - {file_path}")
    if not missing_files and not unreadable_files:
        logger.info("✓ All required data files exist and are readable (or ignorable if empty first line).")
        
    # 3. Check model files (e.g., tokenizer model)
    logger.info("\n3. Verifying model files...")
    missing_model_files = verify_model_files()
    if missing_model_files:
        all_checks_passed = False
        logger.error("✗ Missing model files:")
        for file_path in missing_model_files:
            logger.error(f"  - {file_path}")
    else:
        logger.info("✓ All required model files exist.")
        
    # 4. Check GPU
    logger.info("\n4. Verifying GPU...")
    gpu_ok = verify_gpu()
    if not gpu_ok and torch.cuda.is_available(): # Only an error if CUDA was expected to work but didn't report info
        all_checks_passed = False # This path might not be hit if verify_gpu always returns True
        logger.error("✗ GPU verification reported issues despite CUDA being available.")
    # verify_gpu() logs info but doesn't necessarily make the check fail for CPU-only environments.
    
    # 5. Check model components
    logger.info("\n5. Verifying model components and basic forward pass...")
    model_ok = verify_model_components()
    if not model_ok:
        all_checks_passed = False
        # Detailed error logged in verify_model_components
    
    # Summary
    logger.info("\n=== Verification Summary ===")
    if all_checks_passed:
        logger.info("✓✓✓ All critical checks passed! Your setup seems ready.")
        if not torch.cuda.is_available():
            logger.info("  Reminder: Running in CPU mode. GPU will be needed for efficient training.")
    else:
        logger.error("⚠ Some checks failed. Please review the logs above and address the issues before proceeding.")

if __name__ == '__main__':
    main() 