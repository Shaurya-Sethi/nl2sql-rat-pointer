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
import glob # For finding latest checkpoint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s' # Added logger name
)
logger = logging.getLogger(__name__)

# Global variables for signal handling
global_trainer = None
training_interrupted = False  # Added global flag for signal handling

# Signal handlers for graceful shutdown
def handle_signal(signum, frame):
    """Handle termination signals by setting a flag for graceful shutdown"""
    global training_interrupted
    sig_name = signal.Signals(signum).name
    logger.info(f"{sig_name} received, flagging for graceful shutdown. Training will attempt to save a checkpoint and exit after the current step/batch.")
    training_interrupted = True

# Worker init function for DataLoader
def worker_init_fn(worker_id):
    """Makes worker processes ignore SIGINT. Main process will handle it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    global global_trainer, training_interrupted  # Add the global flag to main scope
    
    parser = argparse.ArgumentParser(description='Train NL2SQL model')
    parser.add_argument('--phase', type=str, required=True, choices=['pretrain', 'sft'],
                      help='Training phase: pretrain or sft')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file (e.g., src/config.yaml)')
    parser.add_argument('--pretrained_model', type=str,
                      help='Path to pretrained model for SFT phase (load from checkpoint if not resuming an SFT run)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                      help='Path to a specific checkpoint file to resume training from. Overrides automatic latest checkpoint detection.')
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

    # Initialize relation matrix builder (only relevant for SFT but init anyway for consistency)
    relation_builder = RelationMatrixBuilder(
        tokenizer=tokenizer,
        num_relations=model_config.num_relations
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
        collate_fn=collate_fn_to_use,
        worker_init_fn=worker_init_fn if model_config.num_workers > 0 else None,
        pin_memory=True if torch.cuda.is_available() else False # Added pin_memory
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=model_config.max_batch_size, # Use max_batch_size for eval
        shuffle=False,
        num_workers=model_config.num_workers,
        collate_fn=collate_fn_to_use,
        worker_init_fn=worker_init_fn if model_config.num_workers > 0 else None,
        pin_memory=True if torch.cuda.is_available() else False # Added pin_memory
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
    
    # Checkpoint loading / Resumption logic
    checkpoint_to_load = None
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            checkpoint_to_load = args.resume_from_checkpoint
            logger.info(f"Attempting to resume from specified checkpoint: {checkpoint_to_load}")
        else:
            logger.warning(f"Specified resume_from_checkpoint not found: {args.resume_from_checkpoint}. Will check for latest_checkpoint.pt or start fresh.")
            args.resume_from_checkpoint = None # Clear it so we try latest

    if not checkpoint_to_load:
        latest_checkpoint_path = Path(model_config.output_dir) / 'latest_checkpoint.pt'
        if latest_checkpoint_path.is_file():
            checkpoint_to_load = str(latest_checkpoint_path)
            logger.info(f"Found latest_checkpoint.pt. Attempting to resume from: {checkpoint_to_load}")
        else:
            logger.info(f"No latest_checkpoint.pt found in {model_config.output_dir}. Will check for SFT pretrained model or start fresh.")

    if checkpoint_to_load:
        try:
            trainer.load_checkpoint(checkpoint_to_load)
            logger.info(f"Successfully resumed training from checkpoint: {checkpoint_to_load}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_to_load}: {e}. Training will start from scratch or from --pretrained_model if SFT.")
            # Reset trainer's epoch and global_step if loading failed mid-way
            trainer.epoch = 0
            trainer.global_step = 0
            trainer.best_val_loss = float('inf')
            # Re-initialize scheduler if checkpoint load failed and might have partially set it
            trainer.scheduler = get_cosine_schedule_with_warmup(
                trainer.optimizer, # Optimizer should be fresh from model
                num_warmup_steps=model_config.warmup_steps,
                num_training_steps=model_config.max_steps
            )
    elif args.phase == 'sft' and args.pretrained_model:
        # This case is for starting SFT from a --pretrained_model (e.g. from pretraining phase),
        # NOT for resuming an SFT run. Resuming SFT is handled by latest_checkpoint.pt or --resume_from_checkpoint.
        try:
            # For loading a model from a different phase (e.g., pretrain model for SFT start)
            # We only load model_state_dict, not optimizer, scheduler, epoch, etc.
            logger.info(f"No SFT checkpoint to resume. Attempting to load weights from --pretrained_model: {args.pretrained_model} for SFT start.")
            sft_initial_checkpoint = torch.load(args.pretrained_model, map_location=device)
            if 'model_state_dict' in sft_initial_checkpoint:
                # Load with strict=False if vocab size changed or other minor architecture diffs expected
                # For example, if pretraining had a different output layer for LM task.
                # However, core transformer blocks should be compatible.
                missing_keys, unexpected_keys = model.load_state_dict(sft_initial_checkpoint['model_state_dict'], strict=False)
                if missing_keys:
                    logger.warning(f"During SFT init from --pretrained_model, some keys were missing in checkpoint: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"During SFT init from --pretrained_model, some keys in checkpoint were unexpected: {unexpected_keys}")
                logger.info(f"Successfully loaded model weights from --pretrained_model: {args.pretrained_model} for SFT start.")
            else:
                logger.warning(f"--pretrained_model specified ({args.pretrained_model}) but it does not contain 'model_state_dict'. Starting SFT from scratch.")

        except Exception as e:
            logger.error(f"Failed to load --pretrained_model {args.pretrained_model} for SFT: {e}. Starting SFT from scratch.")
    else:
        logger.info(f"No checkpoint found to resume from, and not SFT with --pretrained_model. Starting {args.phase} training from scratch.")

    logger.info(f"Starting {args.phase} training. Will train until global_step reaches {model_config.max_steps}.")
    logger.info(f"Initial state: Epoch {trainer.epoch}, Global Step {trainer.global_step}, Best Val Loss {trainer.best_val_loss:.4f}")

    try:
        while trainer.global_step < model_config.max_steps:
            if training_interrupted:
                logger.info("Training interruption flag checked at start of epoch. Breaking loop.")
                break

            logger.info(f"Starting Epoch {trainer.epoch} (Global Step: {trainer.global_step}/{model_config.max_steps})")
            
            # Train one epoch (train_epoch now manages its own progress bar and step increments)
            train_metrics = trainer.train_epoch() # This will iterate through train_loader
            logger.info(f"Epoch {trainer.epoch} Training Summary: Train Loss: {train_metrics.get('train_loss', float('nan')):.4f}")

            if training_interrupted:
                logger.info("Training interrupted after train_epoch. Saving checkpoint and exiting.")
                break # Exit before validation if interrupted

            # Validate
            if trainer.val_dataloader is not None and model_config.eval_file: # Ensure eval_file is configured
                logger.info(f"Starting Validation for Epoch {trainer.epoch} (Global Step: {trainer.global_step})")
                val_metrics = trainer.validate()
                val_loss = val_metrics.get('val_loss', float('inf'))
                logger.info(f"Epoch {trainer.epoch} Validation Summary: Val Loss: {val_loss:.4f}, Val Perplexity: {val_metrics.get('val_perplexity', float('nan')):.4f}, Val Token Acc: {val_metrics.get('val_token_accuracy', float('nan')):.4f}")
                
                # Early stopping and best model saving logic
                if torch.isfinite(torch.tensor(val_loss)) and val_loss < trainer.best_val_loss:
                    logger.info(f"New best validation loss: {val_loss:.4f} (previously {trainer.best_val_loss:.4f}). Saving best model.")
                    trainer.best_val_loss = val_loss
                    trainer.save_checkpoint(is_best=True) # Saves best_model.pt and latest_checkpoint.pt
                    trainer.no_improve_epochs = 0
                else:
                    trainer.no_improve_epochs += 1
                    logger.info(f"Validation loss did not improve. Previous best: {trainer.best_val_loss:.4f}, Current: {val_loss:.4f}. No improvement epochs: {trainer.no_improve_epochs}/{trainer.early_stopping_patience}")
                    if trainer.early_stopping_patience is not None and trainer.no_improve_epochs >= trainer.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {trainer.epoch + 1} epochs due to no improvement for {trainer.no_improve_epochs} validation rounds.")
                        training_interrupted = True # Use the flag to break outer loop
                        break 
            else:
                logger.info(f"Skipping validation for Epoch {trainer.epoch} as no validation dataloader or eval_file is configured.")
                # If no validation, save a regular checkpoint periodically based on save_steps (handled by global_step check later)
                # or after each epoch if save_steps is large.
                if (trainer.epoch +1) % model_config.save_steps == 0 : # Fallback save if no validation
                     logger.info(f"No validation, saving checkpoint after epoch {trainer.epoch} due to save_steps configuration.")
                     trainer.save_checkpoint(is_best=False)


            if training_interrupted:
                logger.info("Training interrupted after validation/early stopping. Saving checkpoint and exiting.")
                break

            # Save regular checkpoint (based on global_step, more fine-grained than by epoch)
            # Trainer.train_epoch increments global_step. save_checkpoint also saves 'latest_checkpoint.pt'
            # The trainer.save_checkpoint in train_epoch or validation (is_best=True) handles step-based saving logic better.
            # Let's ensure a non-best checkpoint is saved if not covered by best-model logic or early stopping.
            # This is typically if (trainer.global_step % model_config.save_steps == 0) but this check might be better inside train_epoch
            # For now, we rely on save_checkpoint(is_best=True) which also updates latest_checkpoint.pt
            # and we'll add a final save outside the loop.

            # Clear CUDA cache to reduce memory fragmentation (if GPU is used)
            if torch.cuda.is_available():
                logger.debug("Clearing CUDA cache to potentially reduce memory fragmentation.")
                torch.cuda.empty_cache()
            
            trainer.epoch += 1 # Increment epoch counter AFTER a full pass (train + val)

        # End of training loop (either completed max_steps or interrupted)
        if training_interrupted:
            logger.info("Training loop exited due to interruption signal.")
        elif trainer.global_step >= model_config.max_steps:
            logger.info(f"Training completed: global_step ({trainer.global_step}) reached max_steps ({model_config.max_steps}).")
        else:
            logger.info("Training loop exited for an unexpected reason.")

    except KeyboardInterrupt: # Should be caught by signal handler, but as a fallback
        logger.info("KeyboardInterrupt caught directly in main loop. Setting interrupt flag.")
        training_interrupted = True
    except Exception as e:
        logger.error(f"Unhandled error during main training loop: {e}", exc_info=True)
        training_interrupted = True # Treat as interruption to save state
    finally:
        logger.info("Exiting training.")
        if hasattr(trainer, 'model') and trainer.model is not None: # Ensure trainer and model exist
            logger.info("Attempting to save final checkpoint...")
            trainer.save_checkpoint(is_best=False) # Save a final checkpoint regardless of why we exited

            # Save final model state dict separately
            final_model_dir = Path(model_config.output_dir) / args.phase
            final_model_dir.mkdir(parents=True, exist_ok=True)
            final_model_path = final_model_dir / f'final_model_state_dict_epoch{trainer.epoch}_step{trainer.global_step}.pt'
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Final model state_dict saved to {final_model_path}")
        else:
            logger.warning("Trainer or model not available, cannot save final model/checkpoint in finally block.")

        if hasattr(trainer, 'writer') and trainer.writer is not None:
            logger.info("Closing TensorBoard writer...")
            trainer._close_writer()
        
        logger.info("Training script finished.")
        if training_interrupted and not isinstance(sys.exc_info()[0], SystemExit):
             sys.exit(130) # Exit code for Ctrl+C

if __name__ == '__main__':
    main() 