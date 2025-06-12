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
import re # For parsing checkpoint filenames
from typing import Optional
from datetime import datetime # Added for unique SFT directory naming

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
    if global_trainer: # Attempt to tell trainer to stop, if possible
        logger.info("Informing trainer about the interruption.")
        # This is a conceptual addition; actual interruption mechanism inside Trainer would be complex.
        # For now, the flag `training_interrupted` is the primary mechanism checked in the main loop.

# Worker init function for DataLoader
def worker_init_fn(worker_id):
    """Makes worker processes ignore SIGINT. Main process will handle it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def find_most_recent_checkpoint(output_dir: str) -> Optional[str]:
    """Scans the output directory for checkpoints and returns the path to the most recent one based on step number."""
    logger.info(f"Scanning for most recent checkpoint in {output_dir} (excluding latest_checkpoint.pt and best_model.pt)...")
    checkpoints_found = []

    # Pattern for current format: checkpoint_epoch{E}_step{S}.pt
    pattern_epoch_step = re.compile(r"checkpoint_epoch(\d+)_step(\d+)\.pt")
    # Pattern for older format: checkpoint-{S}.pt (like checkpoint-217193.pt)
    pattern_step_only = re.compile(r"checkpoint-(\d+)\.pt")

    # Combine glob patterns to find potential checkpoint files
    candidate_files = glob.glob(os.path.join(output_dir, "checkpoint_epoch*_step*.pt")) + \
                      glob.glob(os.path.join(output_dir, "checkpoint-*.pt"))

    for ckpt_file_path in candidate_files:
        filename = os.path.basename(ckpt_file_path)
        
        # Explicitly skip best_model.pt and latest_checkpoint.pt from this scan
        if filename == "latest_checkpoint.pt" or filename == "best_model.pt":
            continue

        epoch = -1 # Default if not found in filename
        step = -1

        match_epoch_step = pattern_epoch_step.match(filename)
        if match_epoch_step:
            try:
                epoch = int(match_epoch_step.group(1))
                step = int(match_epoch_step.group(2))
            except ValueError:
                logger.warning(f"Could not parse epoch/step from {filename}. Skipping.")
                continue
        else:
            match_step_only = pattern_step_only.match(filename)
            if match_step_only:
                try:
                    step = int(match_step_only.group(1))
                except ValueError:
                    logger.warning(f"Could not parse step from {filename}. Skipping.")
                    continue
            else:
                # This file doesn't match known resume-specific checkpoint patterns
                continue 
        
        if step != -1:
            checkpoints_found.append({'path': ckpt_file_path, 'epoch': epoch, 'step': step, 'filename': filename})

    if not checkpoints_found:
        logger.info("No suitable checkpoint files (e.g., checkpoint_epoch*_step*.pt or checkpoint-*.pt) found for automatic resumption.")
        return None

    # Sort by step number (descending), then by epoch number (descending) as a tie-breaker
    checkpoints_found.sort(key=lambda x: (x['step'], x['epoch']), reverse=True)
    
    most_recent = checkpoints_found[0]
    epoch_str = most_recent['epoch'] if most_recent['epoch'] != -1 else 'N/A'
    logger.info(f"Found {len(checkpoints_found)} potential checkpoint(s) to resume from. Most recent by step is: {most_recent['filename']} (Step: {most_recent['step']}, Epoch: {epoch_str})")
    return most_recent['path']

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
    original_config_path = args.config
    try:
        model_config = NL2SQLConfig.from_yaml(original_config_path, args.phase)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading or parsing configuration: {e}")
        sys.exit(1)

    # Determine and set output directory
    with open(original_config_path, 'r') as f_raw_yaml:
        raw_config_yaml = yaml.safe_load(f_raw_yaml)
    base_output_dir_from_yaml = raw_config_yaml.get('paths', {}).get('output_dir', 'outputs')

    checkpoint_to_load = None # Path to the checkpoint for full trainer state resume
    initialize_from_pretrained = None # Path to checkpoint for model weight initialization only

    if args.phase == 'sft':
        if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
            # Case 1: Explicitly resuming an SFT run from a specific SFT checkpoint.
            logger.info(f"SFT: Attempting to explicitly resume from SFT checkpoint: {args.resume_from_checkpoint}")
            model_config.output_dir = str(Path(args.resume_from_checkpoint).parent)
            checkpoint_to_load = args.resume_from_checkpoint
            logger.info(f"SFT: Output directory set to SFT checkpoint's parent: {model_config.output_dir}")
        elif args.pretrained_model:
            # Case 2: Starting a NEW SFT run, using --pretrained_model for initial weights.
            logger.info(f"SFT: Starting new SFT run. Initializing model weights from --pretrained_model: {args.pretrained_model}")
            sft_run_name = f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_config.output_dir = os.path.join(base_output_dir_from_yaml, sft_run_name)
            initialize_from_pretrained = args.pretrained_model
            logger.info(f"SFT: New SFT output directory created: {model_config.output_dir}")
            # Trainer state will be reset later if this path is taken.
        else:
            # Case 3: Implicit SFT resume (from model_config.output_dir if it's a valid SFT dir) OR SFT from scratch.
            # model_config.output_dir is currently from YAML (e.g., "outputs/").
            logger.info(f"SFT: No explicit resume or --pretrained_model. Checking for auto-resume in: {model_config.output_dir}")
            potential_latest_ckpt = Path(model_config.output_dir) / 'latest_checkpoint.pt'
            most_recent_step_ckpt = find_most_recent_checkpoint(model_config.output_dir)

            # Heuristic: if output_dir is the generic base_output_dir_from_yaml (e.g. "outputs")
            # and it contains checkpoints, these are likely pretraining. Don't auto-resume SFT state from them.
            # Only auto-resume if model_config.output_dir is *not* the generic base and contains checkpoints.
            is_generic_output_dir = (Path(model_config.output_dir).resolve() == Path(base_output_dir_from_yaml).resolve())

            if not is_generic_output_dir and (potential_latest_ckpt.is_file() or most_recent_step_ckpt):
                if potential_latest_ckpt.is_file():
                    logger.info(f"SFT: Found latest_checkpoint.pt in specific SFT directory {model_config.output_dir}. Attempting auto-resume.")
                    checkpoint_to_load = str(potential_latest_ckpt)
                elif most_recent_step_ckpt:
                    logger.info(f"SFT: Found step-based checkpoint {most_recent_step_ckpt} in specific SFT directory {model_config.output_dir}. Attempting auto-resume.")
                    checkpoint_to_load = most_recent_step_ckpt
                # model_config.output_dir is already set to the SFT directory.
            else:
                if is_generic_output_dir and (potential_latest_ckpt.is_file() or most_recent_step_ckpt):
                    logger.info(f"SFT: Checkpoints found in generic output directory '{model_config.output_dir}'. These are assumed to be pretraining. "
                                f"To start SFT from these, use --pretrained_model. Starting SFT from scratch or requiring explicit SFT resume.")
                
                logger.info(f"SFT: No resumable SFT checkpoint found in {model_config.output_dir} (or it's a generic dir with pretrain checkpoints). Starting SFT from scratch.")
                sft_run_name = f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                model_config.output_dir = os.path.join(base_output_dir_from_yaml, sft_run_name)
                logger.info(f"SFT: New SFT output directory for 'from scratch' run: {model_config.output_dir}")
                # checkpoint_to_load remains None. Trainer state is fresh.
    else: # Pretrain phase
        # Pretrain always uses the output_dir from config or a default like "outputs/pretrain_..."
        # For simplicity, pretrain output logic can be less complex or use a fixed subfolder if needed.
        # Current logic: model_config.output_dir is used as is from YAML for pretrain.
        logger.info(f"Pretrain phase. Output directory: {model_config.output_dir}")

    Path(model_config.output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for this run: {model_config.output_dir}")

    # Handle pointer_generator for pretraining phase
    if args.phase == 'pretrain' and model_config.use_pointer_generator:
        logger.info("Pointer-generator is enabled for pretraining. PretrainingDataset will supply dummy schema_mask. Ensure compatibility in downstream SFT.")

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
    model = NL2SQLTransformer(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        num_relations=model_config.num_relations,
        dropout=model_config.dropout,
        max_len=model_config.max_len,
        use_pointer_generator=model_config.use_pointer_generator,
        pad_token_id=model_config.pad_token_id,
        cot_start_token_id=tokenizer.get_special_token_id('COT_START'),
        sql_end_token_id=tokenizer.get_special_token_id('SQL_END')
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
            config=model_config  # Pass the full config object
        )
        eval_dataset = SFTDataset(
            data_file=model_config.eval_file,
            tokenizer=tokenizer,
            relation_builder=relation_builder,
            config=model_config  # Pass the full config object
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
        tokenizer=tokenizer,  # Pass tokenizer for periodic generation
    )
    global_trainer = trainer  # Set the global trainer for signal handlers

    # Checkpoint loading / Resumption logic
    if checkpoint_to_load: # This implies resuming a full SFT state or pretrain state
        try:
            trainer.load_checkpoint(checkpoint_to_load) # This loads full state: model, optimizer, scheduler, epoch, step, best_val_loss
            logger.info(f"Successfully resumed training state using checkpoint: {checkpoint_to_load}")
            current_lr = trainer.optimizer.param_groups[0]['lr']
            config_lr = model_config.learning_rate
            logger.info(f"Learning Rate Check - Config: {config_lr}, Current: {current_lr}")
            if current_lr != config_lr:
                logger.warning(f"Learning rate from checkpoint ({current_lr}) differs from config ({config_lr})")
                if args.phase == 'sft': # For SFT, we typically want to enforce the SFT LR from config on resume
                    logger.info("SFT Phase: Updating learning rate to match SFT config.")
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = config_lr
                    # Re-initialize scheduler with the correct SFT LR and current step if SFT
                    # trainer.scheduler might be None if loaded from an old checkpoint or different setup
                    trainer.scheduler = get_cosine_schedule_with_warmup(
                        trainer.optimizer,
                        num_warmup_steps=model_config.warmup_steps,
                        num_training_steps=model_config.max_steps
                    )
                    # Fast-forward scheduler to the current global_step loaded from checkpoint
                    for _ in range(trainer.global_step):
                        trainer.scheduler.step()
                    logger.info(f"SFT Phase: Scheduler re-initialized and fast-forwarded to global_step: {trainer.global_step}")
                else:
                    logger.info("Pretrain Phase: Using learning rate from checkpoint.")

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_to_load}: {e}. Training will start from scratch or from --pretrained_model if SFT.", exc_info=True)
            # Reset trainer state as checkpoint loading failed
            trainer.epoch = 0
            trainer.global_step = 0
            trainer.best_val_loss = float('inf')
            trainer.no_improve_epochs = 0
            if trainer.optimizer is None: raise ValueError("Trainer optimizer is None after failed checkpoint load attempt.")
            trainer.scheduler = get_cosine_schedule_with_warmup(
                trainer.optimizer,
                num_warmup_steps=model_config.warmup_steps,
                num_training_steps=model_config.max_steps
            )
            logger.info("Reinitialized scheduler after failed checkpoint load.")
            checkpoint_to_load = None # Ensure we don't think we loaded something

    # This block is specifically for initializing SFT from a --pretrained_model (e.g. pretrain checkpoint)
    # It runs if checkpoint_to_load was NOT set by SFT resume logic AND initialize_from_pretrained IS set.
    if args.phase == 'sft' and initialize_from_pretrained and not checkpoint_to_load:
        try:
            logger.info(f"SFT: Initializing model weights from --pretrained_model: {initialize_from_pretrained} for a NEW SFT run.")
            sft_initial_checkpoint = torch.load(initialize_from_pretrained, map_location=device)
            if 'model_state_dict' in sft_initial_checkpoint:
                missing_keys, unexpected_keys = model.load_state_dict(sft_initial_checkpoint['model_state_dict'], strict=False)
                if missing_keys: logger.warning(f"SFT init from --pretrained_model: missing keys: {missing_keys}")
                if unexpected_keys: logger.warning(f"SFT init from --pretrained_model: unexpected keys: {unexpected_keys}")
                logger.info(f"SFT: Successfully loaded model weights from --pretrained_model: {initialize_from_pretrained}")

                logger.info("SFT: Resetting trainer state (epoch, step, loss, scheduler) for new SFT run from pretrained model.")
                trainer.epoch = 0
                trainer.global_step = 0
                trainer.best_val_loss = float('inf')
                trainer.no_improve_epochs = 0
                
                if trainer.optimizer is None: raise ValueError("SFT: Trainer optimizer is None before SFT scheduler re-init.")
                trainer.scheduler = get_cosine_schedule_with_warmup(
                    trainer.optimizer,
                    num_warmup_steps=model_config.warmup_steps, # SFT warmup
                    num_training_steps=model_config.max_steps  # SFT max_steps
                )
                logger.info("SFT: Scheduler re-initialized for new SFT phase.")
            else:
                logger.warning(f"SFT: --pretrained_model ({initialize_from_pretrained}) lacks 'model_state_dict'. SFT starting from scratch.")
        except Exception as e:
            logger.error(f"SFT: Failed to load --pretrained_model {initialize_from_pretrained}: {e}. SFT starting from scratch.", exc_info=True)
    
    elif not checkpoint_to_load and not (args.phase == 'sft' and initialize_from_pretrained) :
        logger.info(f"Starting {args.phase} training from scratch (no checkpoint resumed, no --pretrained_model for SFT weights)." )

    # Ensure BnB optimizer state is initialized robustly, especially after checkpoint load or model manipulation
    if model_config.use_8bit_optimizer and hasattr(trainer.optimizer, '_init_group'):
        logger.info("Performing initial robust check and initialization of bitsandbytes optimizer state before training loop...")
        trainer._ensure_bnb_state(trainer.optimizer)

    logger.info(f"Starting {args.phase} training for {model_config.max_steps} steps...")
    steps_per_epoch = len(train_loader) // model_config.gradient_accumulation_steps
    if steps_per_epoch == 0 : steps_per_epoch = 1 # Avoid division by zero for tiny datasets
    num_epochs_for_max_steps = (model_config.max_steps + steps_per_epoch -1) // steps_per_epoch 

    logger.info(f"Calculated num_epochs based on max_steps: {num_epochs_for_max_steps}")
    logger.info(f"Target total epochs for this run: {model_config.epochs}") # New log for total epochs
    
    start_epoch = trainer.epoch # Get start_epoch from trainer, which is 0 or loaded from checkpoint

    try:
        for current_epoch in range(start_epoch, model_config.epochs):
            trainer.epoch = current_epoch # Set trainer's current epoch
            if training_interrupted:
                logger.info("Training interruption flag checked at start of epoch. Breaking loop.")
                break

            logger.info(f"Starting Epoch {trainer.epoch + 1}/{model_config.epochs} (Global Step: {trainer.global_step}) | Max steps for LR scheduler: {model_config.max_steps}")
            
            # Train one epoch (train_epoch now manages its own progress bar and step increments)
            train_metrics = trainer.train_epoch() # This will iterate through train_loader
            logger.info(f"Epoch {trainer.epoch} Training Summary: Train Loss: {train_metrics.get('train_loss', float('nan')):.4f}")

            # Save checkpoint after epoch completion, before validation
            if not training_interrupted: # Only save epoch-end checkpoint if not already interrupted (avoid double save on interrupt)
                logger.info(f"Saving checkpoint at end of epoch {trainer.epoch} (Global step: {trainer.global_step}) before validation.")
                trainer.save_checkpoint(is_best=False)

            # ====== CUSTOM LEARNING RATE SCHEDULE ======
            # This custom schedule is now applied only during pretraining phase.
            if args.phase == 'pretrain':
                if current_epoch == 0:
                    new_lr = 1e-4
                elif current_epoch == 1:
                    new_lr = 3e-5
                else:
                    new_lr = 1e-6

                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"Pretrain Phase: Set learning rate to {new_lr} at end of epoch {current_epoch}")
            # ====== END CUSTOM LR SCHEDULE ======

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
                # If no validation, regular checkpoint saving is now primarily handled by:
                # 1. Step-based saving within trainer.train_epoch().
                # 2. The epoch-end save that occurs just before this validation block (if validation were enabled).
                # No additional save is needed here.

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
            
            # trainer.epoch += 1 # Increment epoch counter AFTER a full pass (train + val) - REMOVED, epoch is managed by the loop

        # End of training loop (either completed max_steps or interrupted)
        if training_interrupted:
            logger.info("Training loop exited due to interruption signal.")
        elif trainer.epoch >= model_config.epochs -1: # Check if all epochs are completed
            logger.info(f"Training completed: {trainer.epoch + 1}/{model_config.epochs} epochs finished. Final global_step: {trainer.global_step}")
        else:
            logger.info("Training loop exited for an unexpected reason before completing all epochs.")

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

            # Save final model state dict separately in the phase-specific subfolder of output_dir
            # The output_dir in model_config should already be phase-specific (e.g., outputs/sft_xxxx)
            final_model_phase_dir = Path(model_config.output_dir) # Already phase specific
            # final_model_phase_dir.mkdir(parents=True, exist_ok=True) # output_dir already created
            
            final_model_filename = f'final_model_state_dict_phase_{args.phase}_epoch{trainer.epoch}_step{trainer.global_step}.pt'
            final_model_path = final_model_phase_dir / final_model_filename
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