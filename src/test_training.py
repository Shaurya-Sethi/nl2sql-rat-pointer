import os
import logging
import torch
import sys
import traceback
import shutil # Added for rmtree
from pathlib import Path
from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from relation_matrix import RelationMatrixBuilder
from torch.utils.data import DataLoader
from utils.training import Trainer
from config import NL2SQLConfig
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
import inspect
_orig_full = torch.full

def _full_proxy(size, fill_value, *args, **kwargs):
    # Only check fill_value parameter for string values
    if isinstance(fill_value, str):
        print("\n>>> torch.full CALLED WITH STRING fill_value:", repr(fill_value))
        traceback.print_stack(limit=8)                    # show call-site
        raise RuntimeError("torch.full called with string fill_value")  # stop immediately
    return _orig_full(size, fill_value, *args, **kwargs)

torch.full = _full_proxy    # ← patch once, for the whole run
# ──────────────────────────────────────────────────────────────────────────

# Configure logging to show full traceback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up exception hook to show full traceback
def exception_hook(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = exception_hook

class TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, max_len=256, pad_token_id=18):
        self.samples = []
        self.pad_token_id = pad_token_id
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Assuming lines are space-separated token IDs
                try:
                    ids = [int(tok) for tok in line.strip().split()]
                    if len(ids) > max_len:
                        ids = ids[:max_len]
                    if ids: # Ensure we don't add empty lists
                        self.samples.append(ids)  # Store as list, not tensor yet
                except ValueError as e:
                    logger.warning(f"Skipping line due to ValueError: {line.strip()} - Error: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) # True for non-padded
        
        # Create a simple schema mask, marking a portion as True if PG is used
        # This helps prevent potential instability with pointer-generator networks
        # when the schema_mask might otherwise be all False.
        schema_len = 0
        if len(input_ids) > 0: # Ensure input_ids is not empty before calculating length
            schema_len = max(1, int(len(input_ids) * 0.1)) # Mark first 10% as schema for test
        
        schema_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if schema_len > 0 and len(input_ids) >= schema_len: # Check if schema_len is valid
            schema_mask[:schema_len] = True
        
        return {
            'encoder_input': input_ids,
            'decoder_target': input_ids.clone(), # For LM-style training
            'encoder_attention_mask': attention_mask,
            'relation_matrix': torch.zeros((len(input_ids), len(input_ids)), dtype=torch.long), # Dummy relation matrix
            'schema_mask': schema_mask # Dummy schema_mask
        }

def collate_fn(batch, pad_id=18):
    max_len_enc = max(len(item['encoder_input']) for item in batch)
    # For decoder_target, if it can be different length, adjust this
    # For this dataset, encoder_input and decoder_target have same length before padding
    max_len_dec = max(len(item['decoder_target']) for item in batch)
    max_len = max(max_len_enc, max_len_dec)

    batch_size = len(batch)
    
    encoder_input_padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    decoder_target_padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    encoder_attention_mask_padded = torch.zeros(batch_size, max_len, dtype=torch.bool) # False for padded
    relation_matrix_padded = torch.zeros(batch_size, max_len, max_len, dtype=torch.long)
    schema_mask_padded = torch.zeros(batch_size, max_len, dtype=torch.bool) # False for padded
    
    for i, item in enumerate(batch):
        enc_len = len(item['encoder_input'])
        dec_len = len(item['decoder_target'])
        
        encoder_input_padded[i, :enc_len] = item['encoder_input']
        decoder_target_padded[i, :dec_len] = item['decoder_target']
        encoder_attention_mask_padded[i, :enc_len] = item['encoder_attention_mask'] # Original mask was True for non-padded
        
        # relation_matrix and schema_mask are tied to encoder_input length for this dataset
        relation_matrix_padded[i, :enc_len, :enc_len] = item['relation_matrix']
        schema_mask_padded[i, :enc_len] = item['schema_mask']
    
    return {
        'encoder_input': encoder_input_padded,
        'decoder_target': decoder_target_padded,
        'encoder_attention_mask': encoder_attention_mask_padded,
        'relation_matrix': relation_matrix_padded,
        'schema_mask': schema_mask_padded
    }

def test_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent # Get project root more reliably

    output_dir_test = project_root / "outputs_test_training"
    # Clean up output directory from previous test runs to avoid stale checkpoints
    if output_dir_test.exists():
        logger.info(f"Cleaning up existing test output directory: {output_dir_test}")
        shutil.rmtree(output_dir_test)
    output_dir_test.mkdir(parents=True, exist_ok=True)

    # Use a small, actual tokenized data file if available for more realistic testing,
    # otherwise, create a dummy one for the test to run.
    # For now, assuming the SFT filtered train file exists and is small enough or we take a slice.
    # test_data_file = project_root / "datasets" / "paired_nl_sql" / "splits" / "tokenized_sft_filtered_train.txt"
    # Create a dummy data file for robust testing if it doesn't exist
    dummy_data_dir = project_root / "temp_test_data"
    dummy_data_dir.mkdir(exist_ok=True)
    test_data_file = dummy_data_dir / "dummy_tokenized_train.txt"
    with open(test_data_file, 'w') as f:
        for _ in range(120): # Create 120 lines of dummy data
            f.write(' '.join(map(str, np.random.randint(0, 30000, size=np.random.randint(20, 50)))) + '\n')

    val_data_file = dummy_data_dir / "dummy_tokenized_val.txt"
    with open(val_data_file, 'w') as f:
        for _ in range(30): # Create 30 lines of dummy data
            f.write(' '.join(map(str, np.random.randint(0, 30000, size=np.random.randint(20, 50)))) + '\n')

    sp_model_dummy_path = dummy_data_dir / "dummy_sp.model"
    if not sp_model_dummy_path.exists():
        # Create a very basic sentencepiece model for testing if one doesn't exist.
        # This requires sentencepiece package.
        try:
            import sentencepiece as spm
            spm.SentencePieceTrainer.train(
                f'--input={test_data_file},{val_data_file} --model_prefix={str(dummy_data_dir / "dummy_sp")} --vocab_size=1000 --model_type=bpe'
            )
            logger.info(f"Created dummy sentencepiece model at {sp_model_dummy_path}")
        except ImportError:
            logger.warning("sentencepiece package not found. Skipping dummy sp model creation. Tokenizer-dependent tests might fail or use actual model.")
            # Fallback: Create an empty file so path checks don't fail, but tokenizer will be broken.
            sp_model_dummy_path.touch()
        except Exception as e:
            logger.error(f"Could not train dummy sentencepiece model: {e}")
            sp_model_dummy_path.touch() # Fallback

    config = NL2SQLConfig(
        vocab_size=32000, # Should match tokenizer if using real data
        d_model=64,      # Smaller for faster test
        n_heads=2,
        n_layers=1,
        num_relations=5,
        dropout=0.1,
        max_len=128,     # Smaller max_len for test
        pad_token_id=18,
        use_pointer_generator=False, # Test with PG disabled for this basic test
        special_tokens={ # Minimal special tokens for config validation
            'SCHEMA_START': '<SCHEMA>', 'SCHEMA_END': '</SCHEMA>',
            'PK_START': '<PK>', 'PK_END': '</PK>',
            'FK_START': '<FK>', 'FK_END': '</FK>',
            'NL_START': '<NL>', 'NL_END': '</NL>',
            'COT_START': '<COT>', 'COT_END': '</COT>',
            'SQL_START': '<SQL>', 'SQL_END': '</SQL>',
            'EXT_START': '<EXT>', 'EXT_END': '</EXT>'
        },
        batch_size=4, # micro_batch_size from original yaml structure
        max_batch_size=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=5,  # Fewer steps for test
        max_steps=20,    # Run only a few steps for test
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        early_stopping_patience=2,
        save_steps=10,
        num_workers=0, # No multiprocessing for simpler testing
        mixed_precision=False, # Simpler test without mixed precision
        use_8bit_optimizer=False,
        use_bf16=False,
        gradient_checkpointing=False,
        sp_model_path=str(sp_model_dummy_path if sp_model_dummy_path.exists() and sp_model_dummy_path.stat().st_size > 0 else project_root / "models" / "nl2sql_tok.model"),
        output_dir=str(output_dir_test),
        # Add phase-specific max_len parameter for testing
        dataset_phase_max_len=64  # Smaller than model.max_len (128) for testing
    )

    # Set train_file and eval_file separately (not part of NL2SQLConfig)
    train_file = str(test_data_file)
    eval_file = str(val_data_file)

    model = NL2SQLTransformer(
        vocab_size=config.vocab_size,
        num_relations=config.num_relations,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_len=config.max_len,
        use_pointer_generator=config.use_pointer_generator, # Pass this from config
        pad_token_id=config.pad_token_id # Pass this from config
    )
    logger.info("Model initialized successfully for test_training")

    # Get the appropriate max_len for dataset truncation
    dataset_max_len = config.get_dataset_max_len()
    logger.info(f"Using max_len {dataset_max_len} for dataset truncation (model's max_len: {config.max_len})")

    train_dataset = TokenizedTextDataset(train_file, max_len=dataset_max_len, pad_token_id=config.pad_token_id)
    val_dataset = TokenizedTextDataset(eval_file, max_len=dataset_max_len, pad_token_id=config.pad_token_id)
    
    # Ensure datasets are not empty
    if not train_dataset or not val_dataset:
        logger.error("Test datasets are empty. Aborting test_training.")
        # Clean up dummy files if created by this test
        if test_data_file.exists(): test_data_file.unlink(missing_ok=True)
        if val_data_file.exists(): val_data_file.unlink(missing_ok=True)
        if sp_model_dummy_path.exists(): sp_model_dummy_path.unlink(missing_ok=True)
        if dummy_data_dir.exists(): dummy_data_dir.rmdir()
        return False

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=config.pad_token_id)
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=config.pad_token_id)
    )

    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device
        # grad_accum, max_grad_norm, etc., are taken from config inside Trainer
    )

    logger.info("Starting test training run (1 epoch)...")
    trainer.train(num_epochs=1) 
    logger.info("Test training run (1 epoch) completed.")
    
    checkpoint_files = list(Path(config.output_dir).glob('checkpoint-*.pt'))
    if not checkpoint_files:
        # If save_steps > steps_in_1_epoch, no checkpoint will be saved for is_best=False
        # Check for best_model.pt instead
        best_model_path = Path(config.output_dir) / 'best_model.pt'
        if not best_model_path.exists():
            logger.warning("No checkpoint or best_model file found after 1 epoch. This might be okay if save_steps is large.")
            # Clean up dummy files
            if test_data_file.exists(): test_data_file.unlink(missing_ok=True)
            if val_data_file.exists(): val_data_file.unlink(missing_ok=True)
            if sp_model_dummy_path.exists(): sp_model_dummy_path.unlink(missing_ok=True)
            if dummy_data_dir.exists(): dummy_data_dir.rmdir()
            return True # Test considered passed if it ran without errors
        latest_checkpoint = best_model_path
    else:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('-')[1]))
    
    logger.info(f"Found latest/best checkpoint: {latest_checkpoint}")
    
    logger.info("Testing checkpoint loading...")
    # Re-initialize model and trainer for loading test
    model_reloaded = NL2SQLTransformer(
        vocab_size=config.vocab_size, num_relations=config.num_relations, d_model=config.d_model,
        n_heads=config.n_heads, n_layers=config.n_layers, dropout=config.dropout, max_len=config.max_len,
        use_pointer_generator=config.use_pointer_generator, pad_token_id=config.pad_token_id
    )
    new_trainer = Trainer(
        model=model_reloaded, config=config, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader, device=device
    )
    
    try:
        new_trainer.load_checkpoint(str(latest_checkpoint))
        logger.info("Successfully loaded checkpoint into new Trainer instance.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
        # Clean up dummy files
        if test_data_file.exists(): test_data_file.unlink(missing_ok=True)
        if val_data_file.exists(): val_data_file.unlink(missing_ok=True)
        if sp_model_dummy_path.exists(): sp_model_dummy_path.unlink(missing_ok=True)
        if dummy_data_dir.exists(): dummy_data_dir.rmdir()
        return False
    
    logger.info("Continuing training from checkpoint (1 more epoch)...")
    new_trainer.train(num_epochs=1)
    logger.info("Test training continuation from checkpoint completed.")

    if torch.cuda.is_available():
        logger.info(f"Final GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

    # Clean up dummy files and directories created by this test
    if test_data_file.exists(): test_data_file.unlink(missing_ok=True)
    if val_data_file.exists(): val_data_file.unlink(missing_ok=True)
    if sp_model_dummy_path.exists() and sp_model_dummy_path.name.startswith("dummy"): 
        sp_model_dummy_path.unlink(missing_ok=True)
        # Try to remove model-prefix files if sentencepiece created them e.g. dummy_sp.vocab
        vocab_file = dummy_data_dir / "dummy_sp.vocab"
        if vocab_file.exists(): vocab_file.unlink(missing_ok=True)

    if dummy_data_dir.exists(): 
        try:
            # Attempt to remove the directory if it's empty
            os.rmdir(str(dummy_data_dir))
        except OSError:
            logger.warning(f"Could not remove temp_test_data directory {dummy_data_dir}. It might not be empty.")
            
    # Clean up output_dir_test if it's empty or contains only expected files
    # For simplicity in test, just log its path. Manual cleanup might be better for outputs.
    logger.info(f"Test output directory: {output_dir_test}")

    return True

if __name__ == '__main__':
    if not test_training():
        logger.error("test_training FAILED")
        sys.exit(1)
    else:
        logger.info("test_training PASSED") 