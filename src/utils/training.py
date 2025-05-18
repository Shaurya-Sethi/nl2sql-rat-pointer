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

logger = logging.getLogger(__name__)

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
        
        # Ensure pad_token_id is 18 everywhere
        assert self.config.pad_token_id == 18, "pad_token_id must be 18 everywhere"
        
        # Initialize optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
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
        
        # Keep track of runtime batch size adjustments
        original_batch_size = self.config.batch_size
        current_batch_size = original_batch_size
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
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
                    logger.info(f"Trimmed batch to {current_batch_size} samples")
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda') if self.mixed_precision else torch.no_grad():
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
                    elif 'log_probs' in outputs:
                        # Pointer-generator output (already in log space)
                        loss = nn.NLLLoss(ignore_index=self.config.pad_token_id)(
                            outputs['log_probs'].view(-1, self.config.vocab_size),
                            batch['decoder_target'][:, 1:].contiguous().view(-1)
                        )
                    else:
                        raise ValueError("Model output must contain either 'logits' or 'log_probs'")
                    
                    # Check for NaN/Inf in loss
                    if not torch.isfinite(loss).all():
                        logger.error(f"Non-finite loss detected: {loss.item()} in batch {batch_idx}, skipping")
                        skipped_batches += 1
                        continue
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Reset consecutive OOM counter since we got past the backward pass
                consecutive_oom = 0
                
                # Check for NaN/Inf in gradients
                has_nan_or_inf_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            logger.error(f"Non-finite gradient detected in {name}, skipping batch {batch_idx}")
                            has_nan_or_inf_grad = True
                            break
                
                if has_nan_or_inf_grad:
                    # Skip this batch due to bad gradients
                    skipped_batches += 1
                    if self.optimizer:
                        self.optimizer.zero_grad()
                    continue
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Additional check for NaN/Inf after unscaling
                    if self.mixed_precision:
                        has_nan_or_inf_grad = False
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if not torch.isfinite(param.grad).all():
                                    logger.error(f"Non-finite gradient detected after unscaling in {name}, skipping")
                                    has_nan_or_inf_grad = True
                                    break
                        
                        if has_nan_or_inf_grad:
                            # Skip this batch due to bad gradients after unscaling
                            skipped_batches += 1
                            self.optimizer.zero_grad()
                            continue
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    if self.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Update metrics
                step_loss = loss.item() * self.gradient_accumulation_steps
                step_losses.append(step_loss)
                total_loss += step_loss
                total_tokens += batch['encoder_attention_mask'].sum().item()
                num_batches += 1
                
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
                        logger.info(f"Training stable, attempting to increase batch size to {new_batch_size}")
                        current_batch_size = new_batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Empty CUDA cache
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        
                    consecutive_oom += 1
                    logger.error(f"GPU OOM in batch {batch_idx}. Consecutive OOMs: {consecutive_oom}")
                    skipped_batches += 1
                    
                    # Implement dynamic batch size reduction
                    if consecutive_oom >= max_consecutive_oom:
                        # Reduce batch size by half (minimum 1)
                        new_batch_size = max(1, current_batch_size // 2)
                        
                        # Only adjust if we can make it smaller
                        if new_batch_size < current_batch_size:
                            logger.warning(
                                f"Too many consecutive OOMs ({consecutive_oom}). "
                                f"Reducing batch size from {current_batch_size} to {new_batch_size}"
                            )
                            current_batch_size = new_batch_size
                            consecutive_oom = 0  # Reset counter
                            
                            # Alert trainer if batch size is critically small
                            if current_batch_size <= 2:
                                logger.warning(
                                    "Batch size reduced to critical level. "
                                    "Consider reducing model size or input sequence length."
                                )
                        else:
                            # If we can't reduce further, signal failure
                            if current_batch_size == 1:
                                logger.error(
                                    "OOM with batch size of 1. Training cannot proceed. "
                                    "Try reducing model size, gradient accumulation, or sequence length."
                                )
                                break  # Exit the training loop
                    
                    # Skip this batch and continue
                    continue
                else:
                    raise e
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                skipped_batches += 1
                continue
                
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.current_loss = avg_loss  # Store the current loss
        
        # Final warning about skipped batches
        if skipped_batches > 0:
            logger.warning(f"Epoch {self.epoch}: Skipped {skipped_batches}/{len(self.train_dataloader)} batches "
                          f"({skipped_batches/len(self.train_dataloader)*100:.1f}%)")
        
        # Log batch size adaptation info
        if current_batch_size != original_batch_size:
            logger.info(f"Batch size adapted from {original_batch_size} to {current_batch_size}")
        
        return {'train_loss': avg_loss}
        
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
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
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
                    elif 'log_probs' in outputs:
                        # Pointer-generator output (already in log space)
                        loss = nn.NLLLoss(ignore_index=self.config.pad_token_id)(
                            outputs['log_probs'].view(-1, self.config.vocab_size),
                            batch['decoder_target'][:, 1:].contiguous().view(-1)
                        )
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
        
        # Final warning about skipped batches
        if skipped_batches > 0:
            logger.warning(f"Validation: Skipped {skipped_batches}/{len(self.val_dataloader)} batches "
                          f"({skipped_batches/len(self.val_dataloader)*100:.1f}%)")
        
        # Update the current loss to validation loss as it's usually a better indicator
        if avg_loss < float('inf') and torch.isfinite(torch.tensor(avg_loss)):
            self.current_loss = avg_loss
        return {'val_loss': avg_loss}
        
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
                'model_config': model_config  # Add model configuration
            }
            
            # Create output directory if it doesn't exist
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regular checkpoint
            checkpoint_path = output_dir / f'checkpoint-{self.global_step}.pt'
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model
            if is_best:
                best_path = output_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model to {best_path}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            # Add NL2SQLConfig to safe globals
            import torch.serialization
            from config import NL2SQLConfig
            torch.serialization.add_safe_globals([NL2SQLConfig])
            
            # Load checkpoint with weights_only=False since we trust our own checkpoints
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Check if model config exists and validate it
            if 'model_config' in checkpoint:
                stored_config = checkpoint['model_config']
                current_config = {
                    'vocab_size': self.config.vocab_size,
                    'd_model': self.config.d_model,
                    'n_heads': self.config.n_heads,
                    'n_layers': self.config.n_layers,
                    'num_relations': self.config.num_relations,
                    'use_pointer_generator': self.config.use_pointer_generator,
                    'pad_token_id': self.config.pad_token_id
                }
                
                # Check for critical mismatches
                critical_mismatch = False
                mismatch_fields = []
                
                for key in ['vocab_size', 'd_model', 'n_heads', 'n_layers', 'use_pointer_generator']:
                    if key in stored_config and stored_config.get(key) != current_config.get(key):
                        mismatch_fields.append(key)
                        if key in ['vocab_size', 'd_model', 'use_pointer_generator']:
                            critical_mismatch = True
                
                if mismatch_fields:
                    mismatch_msg = f"Checkpoint configuration mismatch in fields: {', '.join(mismatch_fields)}"
                    logger.warning(mismatch_msg)
                    
                    if critical_mismatch:
                        raise ValueError(
                            f"Critical model configuration mismatch: {mismatch_msg}. "
                            "Cannot safely load checkpoint with different architecture. "
                            "Use a compatible model configuration or train from scratch."
                        )
                    else:
                        logger.warning(
                            "Non-critical configuration differences detected. "
                            "Continuing with checkpoint loading, but model behavior may be affected."
                        )
            else:
                logger.warning(
                    "Checkpoint does not contain model configuration. "
                    "This may be an older checkpoint format. Loading without verification."
                )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
            
    def train(self, num_epochs: int):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        try:
            for epoch in range(num_epochs):
                self.epoch = epoch
                
                # Train epoch
                train_metrics = self.train_epoch()
                logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")
                
                # Validate
                if self.val_dataloader is not None:
                    val_metrics = self.validate()
                    logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['val_loss']:.4f}")
                    
                    # Early stopping check - ensure loss is finite
                    val_loss = val_metrics['val_loss']
                    if torch.isfinite(torch.tensor(val_loss)) and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                        self.no_improve_epochs = 0
                    else:
                        self.no_improve_epochs += 1
                        if self.no_improve_epochs >= self.early_stopping_patience:
                            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                            break
                        
                # Save regular checkpoint
                if (epoch + 1) % self.config.save_steps == 0:
                    self.save_checkpoint()
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint()
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.save_checkpoint()
            raise 