import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from bitsandbytes.optim import AdamW8bit # type: ignore
from config import NL2SQLConfig
from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from relation_matrix import RelationMatrixBuilder
from utils.training import Trainer
from utils.metrics import compute_metrics # This is a simple acc/loss, not full eval.
from Pretraining_dataset import PretrainingDataset
from SFT_dataset import SFTDataset
import signal  # Added for signal handling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to hold trainer instance for signal handlers
global_trainer = None

# Signal handlers for graceful shutdown
def handle_signal(signum, frame):
    """Handle termination signals by saving checkpoint and exiting gracefully"""
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    logger.info(f"{sig_name} received, saving checkpoint and shutting down gracefully.")
    
    if global_trainer is not None:
        try:
            # Save checkpoint
            global_trainer.save_checkpoint(is_best=False)
            logger.info("Checkpoint saved successfully.")
            
            # Close TensorBoard writer
            global_trainer._close_writer()
            logger.info("TensorBoard writer closed.")
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    else:
        logger.warning(f"{sig_name} received but trainer not initialized yet. Exiting without saving.")
    
    # Exit with success status
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Train NL2SQL model')
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'sft'],
                      help='Training phase: pretrain or sft')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file (e.g., src/config.yaml)')
    parser.add_argument('--pretrained_model', type=str,
                      help='Path to pretrained model for SFT phase (load from checkpoint)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load NL2SQLConfig object
    try:
        model_config = NL2SQLConfig.from_yaml(args.config, args.phase)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading or parsing configuration: {e}")
        sys.exit(1)

    # Handle pointer_generator for pretraining phase
    if args.phase == 'pretrain' and model_config.use_pointer_generator:
        logger.warning(
            "Pointer-generator is enabled in config but current phase is 'pretrain'. "
            "PretrainingDataset does not support schema_mask required by pointer-generator. "
            "Forcing use_pointer_generator to False for pretraining."
        )
        model_config.use_pointer_generator = False

    # Initialize tokenizer
    try:
        tokenizer = NL2SQLTokenizer(model_config.sp_model_path, model_config.special_tokens)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Initialize relation matrix builder
    relation_builder = RelationMatrixBuilder(
        sp_model_path=model_config.sp_model_path,
        special_tokens=model_config.special_tokens,
        num_relations=model_config.num_relations,
        phase_max_len=model_config.phase_max_len if hasattr(model_config, 'phase_max_len') else 1664
    )

    # Initialize model
    if args.phase == 'sft' and args.pretrained_model:
        logger.info(f"Loading pretrained model checkpoint from {args.pretrained_model} for SFT.")
        model = NL2SQLTransformer(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            num_relations=model_config.num_relations,
            dropout=model_config.dropout,
            max_len=model_config.max_len,
            use_pointer_generator=model_config.use_pointer_generator,
            pad_token_id=model_config.pad_token_id
        )
    else:
        model = NL2SQLTransformer(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            num_relations=model_config.num_relations,
            dropout=model_config.dropout,
            max_len=model_config.max_len,
            use_pointer_generator=model_config.use_pointer_generator,
            pad_token_id=model_config.pad_token_id
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Get the appropriate max_len for dataset truncation
    dataset_max_len = model_config.get_dataset_max_len()
    logger.info(f"Using max_len {dataset_max_len} for dataset truncation (model's max_len: {model_config.max_len})")

    if args.phase == 'pretrain':
        train_dataset = PretrainingDataset(
            data_file=model_config.train_file,
            tokenizer=tokenizer,
            max_len=dataset_max_len
        )
        eval_dataset = PretrainingDataset(
            data_file=model_config.eval_file,
            tokenizer=tokenizer,
            max_len=dataset_max_len
        )
        collate_fn_to_use = lambda batch: PretrainingDataset.collate_fn(batch, pad_id=model_config.pad_token_id)
    else:  # sft
        train_dataset = SFTDataset(
            data_file=model_config.train_file,
            tokenizer=tokenizer,
            relation_builder=relation_builder,
            max_len=dataset_max_len,
            pad_token_id=model_config.pad_token_id
        )
        eval_dataset = SFTDataset(
            data_file=model_config.eval_file,
            tokenizer=tokenizer,
            relation_builder=relation_builder,
            max_len=dataset_max_len,
            pad_token_id=model_config.pad_token_id
        )
        collate_fn_to_use = lambda batch: SFTDataset.collate_fn(batch, pad_id=model_config.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=model_config.num_workers,
        collate_fn=collate_fn_to_use
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers,
        collate_fn=collate_fn_to_use
    )

    global global_trainer  # Use the global trainer variable
    trainer = Trainer(
        model=model,
        config=model_config,
        train_dataloader=train_loader,
        val_dataloader=eval_loader,
        device=device,
    )
    global_trainer = trainer  # Set the global trainer for signal handlers

    if args.phase == 'sft' and args.pretrained_model:
        try:
            trainer.load_checkpoint(args.pretrained_model)
            logger.info(f"Successfully loaded weights from checkpoint: {args.pretrained_model}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {args.pretrained_model}: {e}. Starting SFT from scratch.")

    # Register signal handlers
    logger.info("Registering signal handlers for graceful shutdown")
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info(f"Starting {args.phase} training for {model_config.max_steps} steps...")
    steps_per_epoch = len(train_loader) // model_config.gradient_accumulation_steps
    if steps_per_epoch == 0 : steps_per_epoch = 1 # Avoid division by zero for tiny datasets
    num_epochs_for_max_steps = (model_config.max_steps + steps_per_epoch -1) // steps_per_epoch 

    logger.info(f"Calculated num_epochs based on max_steps: {num_epochs_for_max_steps}")
    trainer.train(num_epochs=num_epochs_for_max_steps)

    final_model_dir = Path(model_config.output_dir) / args.phase
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_model_dir / 'final_model_state_dict.pt'
    torch.save(model.state_dict(), final_model_path)
    
    # Explicitly close the TensorBoard writer to ensure all logs are flushed
    if hasattr(trainer, 'writer') and trainer.writer is not None:
        logger.info("Closing TensorBoard writer...")
        trainer._close_writer()
        
    logger.info(f"Training complete. Final model state_dict saved to {final_model_path}")

if __name__ == '__main__':
    main() 