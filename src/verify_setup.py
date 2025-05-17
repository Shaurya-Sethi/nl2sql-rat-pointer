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
    
    project_root = Path(__file__).parent.parent
    missing_dirs = []
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
        else:
            logger.info(f"✓ Directory exists: {dir_path}")
            
    return missing_dirs

def verify_data_files():
    """Verify all required data files exist and are readable."""
    required_files = [
        'datasets/paired_nl_sql/splits/tokenized_sft_filtered_train.txt',
        'datasets/paired_nl_sql/splits/tokenized_sft_filtered_val.txt',
        'datasets/raw_sql/pretraining corpus/splits/wrapped_tokenized_corpus_train.txt',
        'datasets/raw_sql/pretraining corpus/splits/wrapped_tokenized_corpus_val.txt'
    ]
    
    project_root = Path(__file__).parent.parent
    missing_files = []
    unreadable_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            try:
                # Try to read first line
                with open(full_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        unreadable_files.append(f"{file_path} (empty)")
                    else:
                        logger.info(f"✓ File exists and readable: {file_path}")
            except Exception as e:
                unreadable_files.append(f"{file_path} ({str(e)})")
                
    return missing_files, unreadable_files

def verify_model_files():
    """Verify model files exist."""
    required_files = [
        'models/nl2sql_tok.model'
    ]
    
    project_root = Path(__file__).parent.parent
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            logger.info(f"✓ Model file exists: {file_path}")
            
    return missing_files

def verify_gpu():
    """Verify GPU availability and memory."""
    if not torch.cuda.is_available():
        logger.info("ℹ Running in CPU mode (no GPU available)")
        logger.info("  This is expected for local testing")
        logger.info("  GPU will be required for cloud training")
        return True  # Don't treat this as a failure for local testing
        
    gpu_count = torch.cuda.device_count()
    logger.info(f"✓ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
    return True

def verify_model_components():
    """Verify model components can be initialized."""
    try:
        # Initialize tokenizer
        sp_model_path = Path(__file__).parent.parent / "models" / "nl2sql_tok.model"
        if not sp_model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {sp_model_path}")
            
        # Minimal config for testing
        vocab_size = 32000
        d_model = 128
        n_heads = 4
        n_layers = 2
        num_relations = 5
        dropout = 0.1
        max_len = 256
        
        # Initialize model
        model = NL2SQLTransformer(
            vocab_size=vocab_size,
            num_relations=num_relations,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 256
        test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        test_relation = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.long)
        
        outputs = model(
            encoder_input_ids=test_input,
            decoder_input_ids=test_input,
            encoder_relation_ids=test_relation
        )
        
        logger.info("✓ Model components initialized successfully")
        logger.info(f"  Logits shape: {outputs['logits'].shape}")
        logger.info(f"  Encoder output shape: {outputs['encoder_output'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error verifying model components: {str(e)}")
        return False

def main():
    logger.info("Starting comprehensive setup verification...")
    
    # 1. Check directory structure
    logger.info("\n1. Verifying directory structure...")
    missing_dirs = verify_directory_structure()
    if missing_dirs:
        logger.error("✗ Missing directories:")
        for dir_path in missing_dirs:
            logger.error(f"  - {dir_path}")
    else:
        logger.info("✓ All required directories exist")
        
    # 2. Check data files
    logger.info("\n2. Verifying data files...")
    missing_files, unreadable_files = verify_data_files()
    if missing_files:
        logger.error("✗ Missing files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
    if unreadable_files:
        logger.error("✗ Unreadable files:")
        for file_path in unreadable_files:
            logger.error(f"  - {file_path}")
    if not (missing_files or unreadable_files):
        logger.info("✓ All data files exist and are readable")
        
    # 3. Check model files
    logger.info("\n3. Verifying model files...")
    missing_model_files = verify_model_files()
    if missing_model_files:
        logger.error("✗ Missing model files:")
        for file_path in missing_model_files:
            logger.error(f"  - {file_path}")
    else:
        logger.info("✓ All model files exist")
        
    # 4. Check GPU
    logger.info("\n4. Verifying GPU...")
    verify_gpu()  # Don't store return value since we don't treat it as failure
    
    # 5. Check model components
    logger.info("\n5. Verifying model components...")
    model_ok = verify_model_components()
    
    # Summary
    logger.info("\n=== Verification Summary ===")
    if not any([missing_dirs, missing_files, unreadable_files, missing_model_files]):
        logger.info("✓ All files and directories are in place")
    if not torch.cuda.is_available():
        logger.info("ℹ Running in CPU mode (GPU will be required for cloud training)")
    if model_ok:
        logger.info("✓ Model components are working")
        
    if any([missing_dirs, missing_files, unreadable_files, missing_model_files, not model_ok]):
        logger.error("\n⚠ Some checks failed. Please fix the issues above before proceeding to cloud training.")
    else:
        logger.info("\n✓ All checks passed! You're ready for cloud training.")
        if not torch.cuda.is_available():
            logger.info("  Note: You'll need GPU access for actual training.")

if __name__ == '__main__':
    main() 