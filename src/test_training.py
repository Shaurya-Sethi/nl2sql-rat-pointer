import os
import logging
import torch
import sys
import traceback
from pathlib import Path
from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from relation_matrix import RelationMatrixBuilder
from torch.utils.data import DataLoader
from utils.training import Trainer
from config import NL2SQLConfig

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
    def __init__(self, data_file, max_len=256, pad_token_id=0):
        self.samples = []
        self.pad_token_id = pad_token_id
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                ids = [int(tok) for tok in line.strip().split()]
                if len(ids) > max_len:
                    ids = ids[:max_len]
                self.samples.append(ids)  # Store as list, not tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        # Convert to tensor without device
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        return {
            'encoder_input': input_ids,
            'decoder_target': input_ids.clone(),
            'encoder_attention_mask': attention_mask,
            'relation_matrix': torch.zeros((len(input_ids), len(input_ids)), dtype=torch.long)
        }

def collate_fn(batch, pad_id=18):
    max_len = max(len(item['encoder_input']) for item in batch)
    batch_size = len(batch)
    
    # Initialize tensors with pad_id
    encoder_input = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    decoder_target = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    encoder_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    relation_matrix = torch.zeros(batch_size, max_len, max_len, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        length = len(item['encoder_input'])
        encoder_input[i, :length] = item['encoder_input']
        decoder_target[i, :length] = item['decoder_target']
        encoder_attention_mask[i, :length] = item['encoder_attention_mask']
        relation_matrix[i, :length, :length] = item['relation_matrix']
    
    return {
        'encoder_input': encoder_input,
        'decoder_target': decoder_target,
        'encoder_attention_mask': encoder_attention_mask,
        'relation_matrix': relation_matrix
    }

def test_training():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("No GPU available, using CPU instead")

    # Get project root and data files
    project_root = Path(__file__).parent.parent
    train_file = project_root / "datasets" / "paired_nl_sql" / "splits" / "tokenized_sft_filtered_train.txt"
    val_file = project_root / "datasets" / "paired_nl_sql" / "splits" / "tokenized_sft_filtered_val.txt"
    
    if not train_file.exists() or not val_file.exists():
        logger.error(f"Data files not found: {train_file} or {val_file}")
        return False

    # Minimal model config for testing
    config = NL2SQLConfig(
        vocab_size=32000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_relations=5,
        dropout=0.1,
        max_len=256,
        pad_token_id=18,
        special_tokens={
            'SCHEMA_START': '<SCHEMA>',
            'SCHEMA_END': '</SCHEMA>',
            'PK_START': '<PK>',
            'PK_END': '</PK>',
            'FK_START': '<FK>',
            'FK_END': '</FK>',
            'NL_START': '<NL>',
            'NL_END': '</NL>',
            'COT_START': '<COT>',
            'COT_END': '</COT>',
            'SQL_START': '<SQL>',
            'SQL_END': '</SQL>',
            'EXT_START': '<EXT>',
            'EXT_END': '</EXT>'
        },
        batch_size=4,
        max_batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=10,
        max_steps=100,
        gradient_accumulation_steps=2,  # Test gradient accumulation
        max_grad_norm=1.0,
        early_stopping_patience=3,
        save_steps=20,
        num_workers=0,
        mixed_precision=True if torch.cuda.is_available() else False,
        use_8bit_optimizer=False,
        use_bf16=False,
        gradient_checkpointing=False,
        sp_model_path=str(project_root / "models" / "nl2sql_tok.model"),
        output_dir=str(project_root / "outputs")
    )

    # Initialize model
    model = NL2SQLTransformer(
        vocab_size=config.vocab_size,
        num_relations=config.num_relations,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_len=config.max_len
    )
    logger.info("Model initialized successfully")

    # Create datasets and dataloaders
    train_dataset = TokenizedTextDataset(str(train_file), max_len=config.max_len)
    val_dataset = TokenizedTextDataset(str(val_file), max_len=config.max_len)
    
    # Use only first 100 examples for quick testing
    train_dataset.samples = train_dataset.samples[:100]
    val_dataset.samples = val_dataset.samples[:20]  # Smaller validation set
    
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

    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,  # Add validation dataloader
        device=device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        early_stopping_patience=config.early_stopping_patience,
        mixed_precision=config.mixed_precision
    )

    # Test checkpointing
    logger.info("Testing checkpointing...")
    trainer.train(num_epochs=1)  # Train for 1 epoch
    
    # Create output directory if it doesn't exist
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the latest checkpoint path
    checkpoint_files = list(output_dir.glob('checkpoint-*.pt'))
    if not checkpoint_files:
        logger.error("No checkpoints found")
        return False
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('-')[1]))
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    
    # Test loading checkpoint
    logger.info("Testing checkpoint loading...")
    new_trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        early_stopping_patience=config.early_stopping_patience,
        mixed_precision=config.mixed_precision
    )
    
    try:
        new_trainer.load_checkpoint(str(latest_checkpoint))
        logger.info("Successfully loaded checkpoint")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return False
    
    # Continue training
    logger.info("Continuing training from checkpoint...")
    new_trainer.train(num_epochs=2)

    # Log final GPU memory usage if using GPU
    if torch.cuda.is_available():
        logger.info(f"Final GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

    return True

if __name__ == '__main__':
    # Don't catch—let the error surface with full traceback
    test_training() 