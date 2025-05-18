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
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Validate batch
                if not validate_batch(batch, self.config):
                    logger.error("Invalid batch, skipping")
                    continue
                    
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
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
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
                total_loss += loss.item() * self.gradient_accumulation_steps
                total_tokens += batch['encoder_attention_mask'].sum().item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    logger.error(f"GPU OOM in batch {batch_idx}. Skipping batch.")
                    continue
                else:
                    raise e
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
                
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
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
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
                try:
                    # Validate batch
                    if not validate_batch(batch, self.config):
                        logger.error("Invalid batch, skipping")
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
                    
                    # Update metrics
                    total_loss += loss.item()
                    total_tokens += batch['encoder_attention_mask'].sum().item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        logger.error(f"GPU OOM in validation batch {batch_idx}. Skipping batch.")
                        continue
                    else:
                        raise e
                        
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
                    
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'val_loss': avg_loss}
        
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        try:
            checkpoint = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
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
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.scaler and checkpoint['scaler_state_dict']:
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
                    
                    # Early stopping check
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
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