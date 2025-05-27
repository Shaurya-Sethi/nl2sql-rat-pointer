import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
from config import NL2SQLConfig
from .validation import validate_batch
import torch.cuda.amp as amp
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import socket
import re
import random
import contextlib # Added for nullcontext
from .bnb_utils import ensure_bnb_state, cleanup_bnb_step_tensors # Import the new utility

DEBUG_VERBOSE = False  # Set True to enable detailed training debug logs

def debug_log(msg, logger=None):
    """Helper function for debug logging that respects DEBUG_VERBOSE flag"""
    if DEBUG_VERBOSE:
        if logger:
            logger.debug(msg)
        else:
            print(msg)

logger = logging.getLogger(__name__)


def _sanitize_hparams(hparams):
    allowed_types = (int, float, str, bool)
    clean_hparams = {}
    for k, v in hparams.items():
        if isinstance(v, allowed_types):
            clean_hparams[k] = v
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            clean_hparams[k] = v.item()
        elif v is not None:
            clean_hparams[k] = str(v)
    return clean_hparams


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: NL2SQLConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 3,
        mixed_precision: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: NL2SQL model
            config: Model configuration
            train_dataloader: Training dataloader
            val_dataloader: Optional validation dataloader
            optimizer: Optional optimizer
            scheduler: Optional learning rate scheduler
            device: Device to train on ('cuda' or 'cpu')
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            early_stopping_patience: Number of epochs to wait before early stopping
            mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        # Apply phase-specific settings from config
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        self.max_grad_norm = self.config.max_grad_norm
        if self.config.early_stopping_patience is not None:
            self.early_stopping_patience = self.config.early_stopping_patience
        self.mixed_precision = self.config.mixed_precision and torch.cuda.is_available()
        
        # Ensure pad_token_id is 18 everywhere
        assert self.config.pad_token_id == 18, "pad_token_id must be 18 everywhere"
        
        # Initialize optimizer if not provided
        if optimizer is None:
            # Determine optimizer type based on config
            if self.config.use_8bit_optimizer:
                try:
                    from bitsandbytes.optim import AdamW8bit
                    self.optimizer = AdamW8bit(
                        model.parameters(),
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay
                    )
                    logger.info("Using 8-bit AdamW optimizer.")
                except ImportError:
                    logger.warning("bitsandbytes not found, falling back to standard AdamW. "
                                   "To use 8-bit optimizer, please install bitsandbytes.")
                    self.optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay
                    )
            else:
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        
        # Initialize mixed precision scaler
        self.scaler = amp.GradScaler() if self.mixed_precision else None
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.epoch = 0
        self.no_improve_epochs = 0
        self.current_loss = None  # Track the most recent average loss
        
        # Initialize TensorBoard writer
        if self.config.tensorboard_log_dir:
            # Create a unique run directory with timestamp and phase information
            from datetime import datetime
            import socket
            import os
            
            # Get training phase (pretraining or sft)
            phase = getattr(config, 'phase', 'training')
            
            # Format timestamp
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            
            # Create a descriptive run name
            run_name = f"{timestamp}_{phase}_d{config.d_model}_l{config.n_layers}_h{config.n_heads}"
            
            # Add pointer-generator info if enabled
            if config.use_pointer_generator:
                run_name += "_pg"
                
            # Add hostname for distributed training identification
            run_name += f"_{socket.gethostname()}"
            
            # Create the full log directory path
            log_dir = os.path.join(self.config.tensorboard_log_dir, run_name)
            
            # Create directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging enabled. Log directory: {log_dir}")
            
            # Log model architecture graph (if possible)
            try:
                # Create dummy inputs for the model
                batch_size, seq_len = 2, min(32, config.max_len)
                dummy_encoder_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=self.device)
                dummy_decoder_input = torch.randint(0, config.vocab_size, (batch_size, seq_len-1), device=self.device)
                dummy_relation_matrix = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.long, device=self.device)
                dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=self.device)
                dummy_schema_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device) if config.use_pointer_generator else None
                
                # Log the model graph
                self.writer.add_graph(
                    model,
                    input_to_model=(
                        dummy_encoder_input,
                        dummy_decoder_input,
                        dummy_relation_matrix,
                        dummy_attention_mask,
                        dummy_schema_mask
                    )
                )
                logger.info("Added model graph to TensorBoard")
            except Exception as e:
                logger.warning(f"Could not add model graph to TensorBoard: {e}")
            
            # Log detailed hyperparameters
            model_params = sum(p.numel() for p in model.parameters()) / 1e6  # Convert to millions
            
            hparams = {
                # Model architecture
                'model/vocab_size': config.vocab_size,
                'model/d_model': config.d_model,
                'model/n_heads': config.n_heads,
                'model/n_layers': config.n_layers, 
                'model/num_relations': config.num_relations,
                'model/dropout': config.dropout,
                'model/max_len': config.max_len,
                'model/params_M': model_params,
                'model/use_pointer_generator': int(config.use_pointer_generator),
                
                # Training configuration
                'train/batch_size': config.batch_size,
                'train/learning_rate': config.learning_rate,
                'train/weight_decay': config.weight_decay,
                'train/warmup_steps': config.warmup_steps,
                'train/max_steps': config.max_steps,
                'train/grad_accum_steps': config.gradient_accumulation_steps,
                'train/effective_batch': config.batch_size * config.gradient_accumulation_steps,
                'train/max_grad_norm': self.max_grad_norm,
                'train/early_stopping': config.early_stopping_patience is not None,
                
                # Runtime configuration
                'runtime/mixed_precision': int(config.mixed_precision),
                'runtime/gradient_checkpointing': int(config.gradient_checkpointing),
                'runtime/device': self.device,
                'runtime/num_workers': config.num_workers,
                'runtime/phase': phase,
            }
            
            # Add hardware info if available
            if torch.cuda.is_available():
                hparams['hardware/gpu'] = torch.cuda.get_device_name(0)
                hparams['hardware/gpu_count'] = torch.cuda.device_count()
            
            # Log the hyperparameters with an empty metrics dict
            # The actual metrics will be logged during training
            self.writer.add_hparams(_sanitize_hparams(hparams), {})
            
            # Write a text summary of the training configuration
            config_text = f"# NL2SQL Training Configuration\n\n"
            config_text += f"- **Phase**: {phase}\n"
            config_text += f"- **Model Size**: {model_params:.2f}M parameters\n"
            config_text += f"- **Architecture**: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}\n"
            config_text += f"- **Batch Size**: {config.batch_size} (x{config.gradient_accumulation_steps} accumulation = {config.batch_size * config.gradient_accumulation_steps} effective)\n"
            config_text += f"- **Learning Rate**: {config.learning_rate}\n"
            config_text += f"- **Hardware**: {self.device}\n"
            config_text += f"- **Mixed Precision**: {'Enabled' if config.mixed_precision else 'Disabled'}\n"
            config_text += f"- **Gradient Checkpointing**: {'Enabled' if config.gradient_checkpointing else 'Disabled'}\n"
            config_text += f"- **Pointer Generator**: {'Enabled' if config.use_pointer_generator else 'Disabled'}\n"
            
            self.writer.add_text('training_config', config_text)
            
        else:
            self.writer = None
            logger.info("TensorBoard logging disabled")
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        step_losses = []  # Track individual step losses
        skipped_batches = 0  # Track OOM or error skipped batches
        consecutive_oom = 0  # Count consecutive OOM errors
        max_consecutive_oom = 3  # Maximum consecutive OOM errors before reducing batch size
        total_ooms_this_epoch = 0 # Initialize total OOMs for this epoch
        
        # Tracking metrics for TensorBoard
        batch_times = []
        grad_norms = []
        
        # Keep track of runtime batch size adjustments
        original_batch_size = self.config.batch_size
        current_batch_size = original_batch_size
        
        # Metrics for TensorBoard
        token_accuracy_sum = 0
        tokens_processed = 0
        
        # For throughput calculation
        epoch_start_time = time.time()
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx == 0 and self.config.use_8bit_optimizer:
                debug_log(f"Epoch {self.epoch}: Ensuring BnB optimizer state at the start of the epoch (batch_idx 0)...", logger)
                ensure_bnb_state(self.optimizer, device=self.device)

            if batch is None:
                logger.warning(f"Trainer.train_epoch: Epoch {self.epoch}, Batch index {batch_idx}: Received None batch from DataLoader. Skipping.")
                skipped_batches += 1 # Ensure this variable is initialized if used for overall stats
                continue
            batch_start_time = time.time()
            
            try:
                # Validate batch
                if not validate_batch(batch, self.config):
                    logger.error("Invalid batch, skipping")
                    skipped_batches += 1
                    continue
                
                # If running with dynamic batch sizes and the current batch is too large, trim it
                if current_batch_size < batch['encoder_input'].size(0):
                    # Trim batch to current batch size
                    for k in batch:
                        if isinstance(batch[k], torch.Tensor):
                            batch[k] = batch[k][:current_batch_size]
                    debug_log(f"Trimmed batch to {current_batch_size} samples", logger)
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                autocast_context = torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.config.use_bf16 else torch.float16) if self.mixed_precision else contextlib.nullcontext()
                with autocast_context:
                    # Reshape attention mask to (batch_size, seq_len, seq_len)
                    encoder_attention_mask = batch['encoder_attention_mask'].unsqueeze(1).expand(-1, batch['encoder_attention_mask'].size(1), -1)
                    
                    outputs = self.model(
                        encoder_input_ids=batch['encoder_input'],
                        decoder_input_ids=batch['decoder_target'][:, :-1],
                        encoder_relation_ids=batch['relation_matrix'],
                        encoder_attention_mask=encoder_attention_mask,
                        schema_mask=batch.get('schema_mask')  # Pass schema_mask if available
                    )
                    
                    # Calculate loss - handle both standard logits and log_probs
                    if 'logits' in outputs:
                        # Standard decoder output
                        loss = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)(
                            outputs['logits'].view(-1, self.config.vocab_size),
                            batch['decoder_target'][:, 1:].contiguous().view(-1)
                        )
                        
                        # Calculate token accuracy for logging
                        if self.writer and self.global_step % self.config.log_every_n_steps == 0:
                            with torch.no_grad():
                                # Get predictions
                                preds = torch.argmax(outputs['logits'], dim=-1)
                                targets = batch['decoder_target'][:, 1:]
                                
                                # Compute token-level accuracy (ignoring padding)
                                pad_mask = (targets != self.config.pad_token_id)
                                correct = ((preds == targets) & pad_mask).sum().item()
                                total = pad_mask.sum().item()
                                
                                if total > 0:
                                    token_accuracy_sum += correct
                                    tokens_processed += total
                    elif 'log_probs' in outputs:
                        # Pointer-generator output (already in log space)
                        loss = nn.NLLLoss(ignore_index=self.config.pad_token_id)(
                            outputs['log_probs'].view(-1, self.config.vocab_size),
                            batch['decoder_target'][:, 1:].contiguous().view(-1)
                        )
                        
                        # Calculate token accuracy for logging
                        if self.writer and self.global_step % self.config.log_every_n_steps == 0:
                            with torch.no_grad():
                                # Get predictions
                                preds = torch.argmax(outputs['log_probs'], dim=-1)
                                targets = batch['decoder_target'][:, 1:]
                                
                                # Compute token-level accuracy (ignoring padding)
                                pad_mask = (targets != self.config.pad_token_id)
                                correct = ((preds == targets) & pad_mask).sum().item()
                                total = pad_mask.sum().item()
                                
                                if total > 0:
                                    token_accuracy_sum += correct
                                    tokens_processed += total
                    else:
                        raise ValueError("Model output must contain either 'logits' or 'log_probs'")
                    
                    # Check for NaN/Inf in loss
                    if not torch.isfinite(loss).all():
                        logger.error(f"Non-finite loss detected: {loss.item()} in batch {batch_idx} at global step {self.global_step}. Halting training.")
                        self.save_checkpoint()
                        self._close_writer()
                        raise ValueError(f"Non-finite loss encountered in batch {batch_idx} at global step {self.global_step}. Training halted.")
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass is done first
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step logic, only if it's an accumulation boundary
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer) # Unscale before clipping
                    
                    if self._has_nan_or_inf_grad_after_unscale(): # Checks grads after unscale (if MP) or just before clip (if not MP, though helper name implies unscale)
                        logger.error(f"Skipping optimizer step for batch {batch_idx}, global step {self.global_step} due to non-finite gradients.")
                        skipped_batches += 1
                        self.optimizer.zero_grad(set_to_none=True) # Clear grads
                        # If using scaler, it's important to call update to keep its state consistent, even if we skip a step.
                        # This is because unscale_ was called. If step is skipped, update ensures scaler is ready for next valid step.
                        if self.mixed_precision:
                            self.scaler.update()
                        continue # Move to next micro-batch, effectively skipping optimizer step and global_step increment

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Log pre-clip gradient norm if enabled
                    if self.writer and self.config.log_grad_norm:
                        # Note: _compute_grad_norm is called after unscale (if MP) and before clip.
                        unclipped_grad_norm = self._compute_grad_norm()
                        self.writer.add_scalar('training/gradient_norm_pre_clip', unclipped_grad_norm, self.global_step)
                        if self.global_step % (self.config.log_every_n_steps * 5) == 0:
                            self._log_layer_gradient_norms(self.global_step)

                    try:
                        if self.mixed_precision:
                            self.scaler.step(self.optimizer)
                            # self.scaler.update() is now after the try-except block
                        else:
                            self.optimizer.step()
                        
                        # If step was successful, update scaler and zero grads, etc.
                        if self.mixed_precision:
                            self.scaler.update()
                        
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        
                        if self.scheduler is not None:
                            self.scheduler.step()
                            if self.writer:
                                current_lr = self.scheduler.get_last_lr()[0]
                                self.writer.add_scalar('training/learning_rate', current_lr, self.global_step)
                        elif self.writer:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            self.writer.add_scalar('training/learning_rate', current_lr, self.global_step)

                    except (KeyError, TypeError) as e:
                        error_str = str(e).lower() # For easier checking
                        if self.config.use_8bit_optimizer and ("'state1'" in error_str or "integer tensors of a single element can be converted to an index" in error_str or "step" in error_str):
                            logger.warning(
                                f"Optimizer state error ('{e}') during step in Epoch {self.epoch}, Batch {batch_idx}, Global Step {self.global_step}. "
                                f"Ensuring BnB state and skipping this batch's optimizer update."
                            )
                            ensure_bnb_state(self.optimizer, device=self.device) # Attempt to fix optimizer state
                            self.optimizer.zero_grad(set_to_none=True) # Clear potentially problematic grads
                            if self.mixed_precision:
                                # Scaler.step() might not have been called or failed. 
                                # Call update() to ensure scaler's internal state is advanced.
                                self.scaler.update() 
                            skipped_batches += 1
                            continue # Move to the next micro-batch, effectively skipping the optimizer step and global_step increment
                        else:
                            logger.error(f"Unhandled Optimizer/Scheduler Error ({type(e).__name__}): {e}. Global step {self.global_step}", exc_info=True)
                            self.save_checkpoint(is_best=False)
                            self._close_writer()
                            raise
                    except RuntimeError as e:
                         # Catch other runtime errors from optimizer.step() or scaler operations
                        logger.error(f"RuntimeError during optimizer step or scaler update: {e}. Global step {self.global_step}", exc_info=True)
                        self.save_checkpoint(is_best=False)
                        self._close_writer()
                        raise
                    finally:
                        # This ensures that even if we `continue` above due to a caught BnB error,
                        # if it was NOT an accumulation step, gradients are cleared.
                        # However, if it WAS an accumulation step and we skipped, zero_grad was already called.
                        # The logic here is tricky. Let's simplify: zero_grad is handled if step succeeds or if BnB error occurs.
                        # If other error, it's raised. If it *wasn't* an accumulation step, grads just accumulate.
                        pass # zero_grad is handled within the success path or the BnB error path of an accumulation step.

                # Update metrics
                step_loss = loss.item() * self.gradient_accumulation_steps
                step_losses.append(step_loss)
                total_loss += step_loss
                total_tokens += batch['encoder_attention_mask'].sum().item()
                num_batches += 1
                
                # Log to TensorBoard
                if self.writer and self.global_step % self.config.log_every_n_steps == 0:
                    # Log loss
                    self.writer.add_scalar('training/step_loss', step_loss, self.global_step)
                    
                    # Log perplexity (exp of loss)
                    self.writer.add_scalar('training/perplexity', torch.exp(torch.tensor(step_loss)).item(), self.global_step)
                    
                    # Log token accuracy if computed
                    if tokens_processed > 0:
                        token_accuracy = token_accuracy_sum / tokens_processed
                        self.writer.add_scalar('training/token_accuracy', token_accuracy, self.global_step)
                        # Reset counters
                        token_accuracy_sum = 0
                        tokens_processed = 0
                    
                    # Log batch time
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    self.writer.add_scalar('performance/batch_time_sec', batch_time, self.global_step)
                    
                    # Log throughput (tokens/sec)
                    if batch_time > 0:
                        tokens_in_batch = batch['encoder_attention_mask'].sum().item()
                        tokens_per_sec = tokens_in_batch / batch_time
                        self.writer.add_scalar('performance/tokens_per_sec', tokens_per_sec, self.global_step)
                    
                    # Log batch size
                    self.writer.add_scalar('training/batch_size', current_batch_size, self.global_step)
                    
                    # Log memory usage if enabled
                    if self.config.log_memory and torch.cuda.is_available():
                        allocated_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                        reserved_memory_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                        max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                        self.writer.add_scalar('performance/gpu_allocated_mb', allocated_memory_mb, self.global_step)
                        self.writer.add_scalar('performance/gpu_reserved_mb', reserved_memory_mb, self.global_step)
                        self.writer.add_scalar('performance/gpu_max_allocated_mb', max_memory_mb, self.global_step)
                        
                        # Log GPU utilization if nvml is available
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            self.writer.add_scalar('performance/gpu_utilization', info.gpu, self.global_step)
                            self.writer.add_scalar('performance/gpu_memory_utilization', info.memory, self.global_step)
                        except:
                            pass  # Skip if pynvml not available
                    
                    # Log histograms if enabled
                    if self.config.log_grad_histogram and self.global_step % (self.config.log_every_n_steps * 10) == 0:
                        # Log parameter histograms
                        self._log_parameter_histograms(self.global_step)
                        
                        # Log activation distributions for key layers if outputs are available
                        if 'encoder_output' in outputs:
                            self.writer.add_histogram('activations/encoder_output', 
                                                   outputs['encoder_output'].detach().float().cpu(), 
                                                   self.global_step)
                            
                        # Log attention weights if available in model layers
                        self._log_attention_weights(self.global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_loss:.4f}",
                    'bs': current_batch_size,
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # If we've been running with reduced batch size and things are going well,
                # we can try to increase batch size back toward the original
                if current_batch_size < original_batch_size and batch_idx % 10 == 0:
                    new_batch_size = min(current_batch_size + 1, original_batch_size)
                    if new_batch_size > current_batch_size:
                        debug_log(f"Training stable, attempting to increase batch size to {new_batch_size}", logger)
                        current_batch_size = new_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available() and hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    
                    total_ooms_this_epoch += 1
                    skipped_batches += 1 
                    consecutive_oom += 1

                    logger.error(
                        f"GPU OOM in Epoch {self.epoch}, Batch {batch_idx} (Global Step: {self.global_step}). "
                        f"Current Batch Size: {current_batch_size}. Consecutive OOMs for this batch size: {consecutive_oom}. "
                        f"Total OOMs this epoch: {total_ooms_this_epoch}. Error: {str(e)}"
                    )

                    if current_batch_size == 1:
                        logger.critical(
                            f"FATAL: GPU OOM with batch size 1 in Epoch {self.epoch}, Batch {batch_idx} (Global Step: {self.global_step}). "
                            "Training cannot continue. Saving state and aborting."
                        )
                        self.save_checkpoint(is_best=False)
                        if self.writer: # Log OOM count before closing writer
                            self.writer.add_scalar('performance/epoch_total_ooms', total_ooms_this_epoch, self.epoch)
                        self._close_writer()
                        raise RuntimeError(
                            f"Unrecoverable GPU OOM with batch size 1 at Epoch {self.epoch}, Global Step {self.global_step}. Training halted."
                        )
                    
                    if total_ooms_this_epoch > 50 and total_ooms_this_epoch % 10 == 0:
                         logger.warning(
                            f"High OOM count in Epoch {self.epoch}: {total_ooms_this_epoch} OOMs. "
                            f"Training stability potentially compromised. Current batch size: {current_batch_size}."
                        )

                    if consecutive_oom >= max_consecutive_oom:
                        new_batch_size = max(1, current_batch_size // 2)
                        if new_batch_size < current_batch_size:
                            logger.warning(
                                f"Reducing batch size due to {consecutive_oom} consecutive OOMs: {current_batch_size} -> {new_batch_size} "
                                f"in Epoch {self.epoch}, Batch {batch_idx} (Global Step: {self.global_step})."
                            )
                            current_batch_size = new_batch_size
                            consecutive_oom = 0 # Reset for the new batch size
                            if current_batch_size == 1:
                                logger.warning(
                                     f"Batch size dynamically reduced to 1. Next OOM will be fatal. Epoch {self.epoch}, Global Step {self.global_step}."
                                )
                    
                    continue # Skip this batch
                
                else: # Handle other RuntimeErrors that are not OOM
                    logger.error(
                        f"Unhandled RuntimeError encountered in Epoch {self.epoch}, Batch {batch_idx} (Global Step: {self.global_step}). "
                        f"Error: {str(e)}. Saving state and aborting training.",
                        exc_info=True
                    )
                    self.save_checkpoint(is_best=False)
                    self._close_writer()
                    raise # Re-raise the original error to halt training
                    
            except Exception as e: # Catches other non-RuntimeErrors (e.g., from data loading, custom logic not covered by NaN checks)
                                   # Note: ValueError from NaN loss check already saves/closes/raises.
                logger.warning(
                    f"Unhandled non-RuntimeError in Epoch {self.epoch}, Batch {batch_idx} (Global Step: {self.global_step}). "
                    f"Skipping batch. Error: {str(e)}",
                    exc_info=True
                )
                skipped_batches += 1
                if self.optimizer: # Ensure grads are cleared if an error happened before optimizer step
                    self.optimizer.zero_grad(set_to_none=True) 
                continue
                
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.current_loss = avg_loss  # Store the current loss
        self.current_train_loss = avg_loss  # Store specifically as train loss
        
        # Calculate epoch metrics
        epoch_duration = time.time() - epoch_start_time
        samples_processed = num_batches * current_batch_size
        
        # Final warning about skipped batches
        if skipped_batches > 0:
            logger.warning(f"Epoch {self.epoch}: Skipped {skipped_batches}/{len(self.train_dataloader)} batches "
                          f"({skipped_batches/len(self.train_dataloader)*100:.1f}%)")
        
        # Log batch size adaptation info
        if current_batch_size != original_batch_size:
            logger.info(f"Batch size adapted from {original_batch_size} to {current_batch_size}")
        
        # Log epoch metrics to TensorBoard
        if self.writer:
            self.writer.add_scalar('training/epoch_loss', avg_loss, self.epoch)
            self.writer.add_scalar('training/epoch_perplexity', torch.exp(torch.tensor(avg_loss)).item(), self.epoch)
            self.writer.add_scalar('performance/epoch_total_ooms', total_ooms_this_epoch, self.epoch) # Log total OOMs for the epoch
            
            # Log additional metrics
            if batch_times:
                avg_batch_time = sum(batch_times) / len(batch_times)
                self.writer.add_scalar('performance/avg_batch_time_sec', avg_batch_time, self.epoch)
                
            if grad_norms:
                avg_grad_norm = sum(grad_norms) / len(grad_norms)
                self.writer.add_scalar('training/avg_grad_norm', avg_grad_norm, self.epoch)
                
            # Log epoch-level performance metrics
            if epoch_duration > 0:
                samples_per_sec = samples_processed / epoch_duration
                self.writer.add_scalar('performance/samples_per_sec', samples_per_sec, self.epoch)
                
                tokens_per_sec = total_tokens / epoch_duration
                self.writer.add_scalar('performance/epoch_tokens_per_sec', tokens_per_sec, self.epoch)
                
            # Log histogram of batch loss distribution
            self.writer.add_histogram('training/batch_loss_distribution', 
                                   torch.tensor(step_losses), 
                                   self.epoch)
                
            # Save a checkpoint of the writer to flush logs
            if self.writer:
                self.writer.flush()
        
        # Clear CUDA cache to reduce memory fragmentation (if GPU is used)
        if torch.cuda.is_available():
            debug_log("Clearing CUDA cache to potentially reduce memory fragmentation.", logger)
            torch.cuda.empty_cache()
        
        return {'train_loss': avg_loss}
        
    def _log_parameter_histograms(self, step: int):
        """Log histograms of model parameters and their gradients."""
        if not self.writer:
            return
            
        # Group parameters by module type for cleaner visualization
        encoder_params = {}
        decoder_params = {}
        other_params = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_params[name] = param
                elif 'decoder' in name:
                    decoder_params[name] = param
                else:
                    other_params[name] = param
        
        # Log encoder parameters
        for name, param in encoder_params.items():
            self.writer.add_histogram(f'encoder_params/{name}', param.data.detach().cpu(), step)
            if param.grad is not None:
                self.writer.add_histogram(f'encoder_grads/{name}', param.grad.detach().cpu(), step)
        
        # Log decoder parameters
        for name, param in decoder_params.items():
            self.writer.add_histogram(f'decoder_params/{name}', param.data.detach().cpu(), step)
            if param.grad is not None:
                self.writer.add_histogram(f'decoder_grads/{name}', param.grad.detach().cpu(), step)
        
        # Log other parameters
        for name, param in other_params.items():
            self.writer.add_histogram(f'other_params/{name}', param.data.detach().cpu(), step)
            if param.grad is not None:
                self.writer.add_histogram(f'other_grads/{name}', param.grad.detach().cpu(), step)
    
    def _log_layer_gradient_norms(self, step: int):
        """Log gradient norms for different layers of the model."""
        if not self.writer:
            return
            
        # Categorize parameters by layer
        encoder_layer_grads = {}
        decoder_layer_grads = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.detach().norm(2).item()
                
                # Identify layer number if possible
                layer_match = re.search(r'layers\.(\d+)', name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    
                    if 'encoder' in name:
                        if layer_num not in encoder_layer_grads:
                            encoder_layer_grads[layer_num] = []
                        encoder_layer_grads[layer_num].append(param_norm)
                    elif 'decoder' in name:
                        if layer_num not in decoder_layer_grads:
                            decoder_layer_grads[layer_num] = []
                        decoder_layer_grads[layer_num].append(param_norm)
        
        # Calculate and log average gradient norm per layer
        for layer_num, norms in encoder_layer_grads.items():
            avg_norm = sum(norms) / len(norms)
            self.writer.add_scalar(f'gradients/encoder_layer_{layer_num}', avg_norm, step)
        
        for layer_num, norms in decoder_layer_grads.items():
            avg_norm = sum(norms) / len(norms)
            self.writer.add_scalar(f'gradients/decoder_layer_{layer_num}', avg_norm, step)
    
    def _log_attention_weights(self, step: int):
        """Log attention weight patterns if accessible in the model."""
        if not self.writer or not hasattr(self.model, 'encoder') or not hasattr(self.model, 'decoder'):
            return
            
        # This is a more advanced feature that requires attention weights to be exposed
        # It would need to be customized based on the specific architecture
        # Here's a placeholder example:
        try:
            # Get the last attention layer in encoder
            # This depends on the model structure and how it exposes attention
            encoder_layers = getattr(self.model.encoder, 'layers', None)
            if encoder_layers and len(encoder_layers) > 0:
                # If the model captures attention weights
                if hasattr(encoder_layers[-1], 'self_attn') and hasattr(encoder_layers[-1].self_attn, 'attn_weights'):
                    attn_weights = encoder_layers[-1].self_attn.attn_weights
                    if attn_weights is not None:
                        self.writer.add_histogram('attention/encoder_last_layer', 
                                               attn_weights.detach().float().cpu(), 
                                               step)
        except Exception as e:
            # Just silently fail, as this is an optional feature
            pass
        
    def _compute_grad_norm(self):
        """Compute total gradient norm for logging."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
        
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        val_step_losses = []  # Track validation step losses
        skipped_batches = 0   # Track skipped batches
        
        # Token accuracy metrics
        token_accuracy_sum = 0
        tokens_processed = 0
        
        validation_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
                if batch is None:
                    logger.warning(f"Trainer.validate: Epoch {self.epoch}, Val Batch index {batch_idx}: Received None batch from DataLoader. Skipping.")
                    skipped_batches += 1 # Ensure this variable is initialized
                    continue
                try:
                    # Validate batch
                    if not validate_batch(batch, self.config):
                        logger.error("Invalid batch, skipping")
                        skipped_batches += 1
                        continue
                        
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Reshape attention mask to (batch_size, seq_len, seq_len)
                    encoder_attention_mask = batch['encoder_attention_mask'].unsqueeze(1).expand(-1, batch['encoder_attention_mask'].size(1), -1)
                    
                    # Forward pass
                    outputs = self.model(
                        encoder_input_ids=batch['encoder_input'],
                        decoder_input_ids=batch['decoder_target'][:, :-1],
                        encoder_relation_ids=batch['relation_matrix'],
                        encoder_attention_mask=encoder_attention_mask,
                        schema_mask=batch.get('schema_mask')  # Pass schema_mask if available
                    )
                    
                    # Calculate loss - handle both standard logits and log_probs
                    if 'logits' in outputs:
                        # Standard decoder output
                        loss = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)(
                            outputs['logits'].view(-1, self.config.vocab_size),
                            batch['decoder_target'][:, 1:].contiguous().view(-1)
                        )
                        
                        # Calculate token accuracy
                        # Get predictions
                        preds = torch.argmax(outputs['logits'], dim=-1)
                        targets = batch['decoder_target'][:, 1:]
                        
                        # Compute token-level accuracy (ignoring padding)
                        pad_mask = (targets != self.config.pad_token_id)
                        correct = ((preds == targets) & pad_mask).sum().item()
                        total = pad_mask.sum().item()
                        
                        if total > 0:
                            token_accuracy_sum += correct
                            tokens_processed += total
                    elif 'log_probs' in outputs:
                        # Pointer-generator output (already in log space)
                        loss = nn.NLLLoss(ignore_index=self.config.pad_token_id)(
                            outputs['log_probs'].view(-1, self.config.vocab_size),
                            batch['decoder_target'][:, 1:].contiguous().view(-1)
                        )
                        
                        # Calculate token accuracy
                        # Get predictions
                        preds = torch.argmax(outputs['log_probs'], dim=-1)
                        targets = batch['decoder_target'][:, 1:]
                        
                        # Compute token-level accuracy (ignoring padding)
                        pad_mask = (targets != self.config.pad_token_id)
                        correct = ((preds == targets) & pad_mask).sum().item()
                        total = pad_mask.sum().item()
                        
                        if total > 0:
                            token_accuracy_sum += correct
                            tokens_processed += total
                    else:
                        raise ValueError("Model output must contain either 'logits' or 'log_probs'")
                    
                    # Check for NaN/Inf in loss
                    if not torch.isfinite(loss).all():
                        logger.error(f"Non-finite validation loss detected: {loss.item()} in batch {batch_idx}, skipping")
                        skipped_batches += 1
                        continue
                    
                    # Update metrics
                    val_step_loss = loss.item()
                    val_step_losses.append(val_step_loss)
                    total_loss += val_step_loss
                    total_tokens += batch['encoder_attention_mask'].sum().item()
                    num_batches += 1
                    
                    # Log validation batch loss to TensorBoard
                    if self.writer and (batch_idx % (self.config.log_every_n_steps * 2) == 0):
                        self.writer.add_scalar('validation/step_loss', val_step_loss, self.global_step + batch_idx)
                        
                        # Log validation perplexity
                        perplexity = torch.exp(torch.tensor(val_step_loss)).item()
                        self.writer.add_scalar('validation/step_perplexity', perplexity, self.global_step + batch_idx)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        logger.error(f"GPU OOM in validation batch {batch_idx}. Skipping batch.")
                        skipped_batches += 1
                        continue
                    else:
                        raise e
                        
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    skipped_batches += 1
                    continue
                    
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Calculate token accuracy
        token_accuracy = token_accuracy_sum / tokens_processed if tokens_processed > 0 else 0.0
        
        # Calculate validation time
        validation_time = time.time() - validation_start_time
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 30 else float('inf')
        
        # Final warning about skipped batches
        if skipped_batches > 0:
            logger.warning(f"Validation: Skipped {skipped_batches}/{len(self.val_dataloader)} batches "
                          f"({skipped_batches/len(self.val_dataloader)*100:.1f}%)")
        
        # Update the current loss to validation loss as it's usually a better indicator
        if avg_loss < float('inf') and torch.isfinite(torch.tensor(avg_loss)):
            self.current_loss = avg_loss
        
        # Log validation metrics to TensorBoard
        if self.writer:
            # Epoch-level metrics
            self.writer.add_scalar('validation/epoch_loss', avg_loss, self.epoch)
            self.writer.add_scalar('validation/perplexity', perplexity, self.epoch)
            self.writer.add_scalar('validation/token_accuracy', token_accuracy, self.epoch)
            self.writer.add_scalar('validation/time_sec', validation_time, self.epoch)
            
            # Throughput metrics
            if total_tokens > 0 and validation_time > 0:
                tokens_per_sec = total_tokens / validation_time
                samples_per_sec = num_batches * batch['encoder_input'].size(0) / validation_time
                self.writer.add_scalar('validation/tokens_per_sec', tokens_per_sec, self.epoch)
                self.writer.add_scalar('validation/samples_per_sec', samples_per_sec, self.epoch)
                
            # Add histograms for validation distribution (once per epoch)
            if val_step_losses:
                try:
                    import numpy as np
                    self.writer.add_histogram('validation/loss_distribution', 
                                            np.array(val_step_losses), 
                                            self.epoch)
                except ImportError:
                    pass
                    
            # Compare training and validation loss
            if hasattr(self, 'current_train_loss') and self.current_train_loss is not None:
                train_val_ratio = self.current_train_loss / avg_loss if avg_loss > 0 else float('inf')
                self.writer.add_scalar('metrics/train_validation_loss_ratio', train_val_ratio, self.epoch)
                
                # Log absolute difference
                loss_diff = abs(self.current_train_loss - avg_loss)
                self.writer.add_scalar('metrics/train_validation_loss_diff', loss_diff, self.epoch)
            
            # Flush TensorBoard logs
            self.writer.flush()
                
        metrics = {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
            'val_token_accuracy': token_accuracy
        }
        
        return metrics

    def _has_nan_or_inf_grad(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    logger.error(f"Non-finite gradient detected in {name} at step {self.global_step}.")
                    return True
        return False

    def _has_nan_or_inf_grad_after_unscale(self):
        """Helper to check for NaN/Inf gradients specifically after scaler.unscale_()."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    logger.error(f"Non-finite gradient detected after unscaling in {name}, global step {self.global_step}.")
                    return True
        return False

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        try:
            # Create a checkpoint version identifier
            config_hash = hash(frozenset({
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'num_relations': self.config.num_relations,
                'use_pointer_generator': self.config.use_pointer_generator,
                'pad_token_id': self.config.pad_token_id
            }.items()))
            
            # Store key model architecture parameters to validate on load
            model_config = {
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'num_relations': self.config.num_relations,
                'use_pointer_generator': self.config.use_pointer_generator,
                'pad_token_id': self.config.pad_token_id,
                'version_hash': config_hash
            }
            
            checkpoint = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                'model_config': model_config,  # Add model configuration
                # RNG states for reproducibility
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate()
            }
            
            # Create output directory if it doesn't exist
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regular checkpoint
            checkpoint_filename = f'checkpoint_epoch{self.epoch}_step{self.global_step}.pt'
            checkpoint_path = output_dir / checkpoint_filename
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Save a copy as latest_checkpoint.pt for easy resumption
            latest_checkpoint_path = output_dir / 'latest_checkpoint.pt'
            torch.save(checkpoint, latest_checkpoint_path)
            logger.info(f"Updated latest_checkpoint.pt to {checkpoint_filename}")
            
            # Save best model
            if is_best:
                best_path = output_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and restore training state."""
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Validate model architecture compatibility
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                # ... existing code ...
            else:
                logger.warning(
                    "Checkpoint does not contain model configuration. "
                    "This may be an older checkpoint format. Cannot validate architecture compatibility. "
                    "Proceed with caution as loading may fail or produce unexpected results."
                )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Store the config's learning rate before loading scheduler state
            config_lr = self.config.learning_rate
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            
            # Ensure learning rate matches config after loading checkpoint
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr != config_lr:
                logger.warning(f"Learning rate from checkpoint ({current_lr}) differs from config ({config_lr}). Updating to match config.")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config_lr
                # If using a scheduler, make sure it's aware of the correct learning rate
                if self.scheduler:
                    self.scheduler.base_lrs = [config_lr] * len(self.optimizer.param_groups)
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(f"Resuming from Epoch: {self.epoch}, Global Step: {self.global_step}, Best Val Loss: {self.best_val_loss:.4f}, Learning Rate: {config_lr}")

            if self.config.use_8bit_optimizer:
                logger.info("Ensuring bitsandbytes optimizer state is robustly initialized after loading checkpoint...")
                ensure_bnb_state(self.optimizer, device=self.device)
                logger.info("Cleaning up any tensor 'step' entries in bitsandbytes optimizer state...")
                cleanup_bnb_step_tensors(self.optimizer)

            # Restore RNG states for reproducibility
            logger.info("Attempting to restore RNG states from checkpoint for reproducibility...")
            rng_keys_to_restore = {
                'torch_rng_state': (lambda state: torch.set_rng_state(state), "PyTorch CPU RNG state"),
                'numpy_rng_state': (lambda state: np.random.set_state(state), "NumPy RNG state"),
                'python_rng_state': (lambda state: random.setstate(state), "Python random module RNG state")
            }
            if torch.cuda.is_available():
                # Only attempt to restore CUDA RNG if CUDA is currently available
                rng_keys_to_restore['cuda_rng_state'] = (lambda state: torch.cuda.set_rng_state(state), "PyTorch CUDA RNG state")

            all_expected_rng_keys_restored_or_handled = True
            for key, (setter, name) in rng_keys_to_restore.items():
                if key in checkpoint:
                    if key == 'cuda_rng_state' and checkpoint[key] is None:
                        # This case means CUDA was not available or not used during checkpoint saving.
                        # If CUDA is available now, its RNG state is not being restored from a None state.
                        logger.info(f"  - {name} was None in checkpoint (e.g., saved on CPU-only or CUDA not used). Current CUDA RNG state preserved.")
                    else:
                        try:
                            setter(checkpoint[key])
                            logger.info(f"  - Successfully restored {name}.")
                        except Exception as e:
                            logger.error(f"  - Failed to restore {name}: {e}")
                            all_expected_rng_keys_restored_or_handled = False
                else:
                    logger.warning(f"  - {name} not found in checkpoint. This might affect exact reproducibility if loading an older checkpoint.")
                    all_expected_rng_keys_restored_or_handled = False
            
            if not all_expected_rng_keys_restored_or_handled:
                logger.warning("One or more RNG states were missing or failed to restore from the checkpoint. Full reproducibility might be affected if this checkpoint is from an older version or a different environment.")
            else:
                logger.info("All available and expected RNG states successfully restored or appropriately handled from checkpoint.")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True) # Added exc_info for better debugging
            raise
        finally:
            # Ensure TensorBoard writer is always closed
            if self.writer is not None: # Check if writer exists before closing
                logger.info("Ensuring TensorBoard writer is closed.")
                self._close_writer()
            
    def _close_writer(self):
        """Close TensorBoard writer to ensure all logs are flushed."""
        if self.writer:
            logger.info("Closing TensorBoard writer")
            self.writer.close()
            self.writer = None 