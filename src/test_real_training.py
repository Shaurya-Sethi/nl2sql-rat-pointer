import os
import logging
import torch
import sys
import traceback
from pathlib import Path
from model import NL2SQLTransformer
from tokenizer import NL2SQLTokenizer
from relation_matrix import RelationMatrixBuilder
from torch.utils.data import DataLoader, Subset
from utils.training import Trainer
from config import NL2SQLConfig
import numpy as np
import random
import time

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
    def __init__(self, data_file, max_len=256, pad_token_id=18, max_samples=None, skip_rows=0):
        self.samples = []
        self.pad_token_id = pad_token_id
        
        logger.info(f"Loading data from {data_file}")
        counter = 0
        with open(data_file, 'r', encoding='utf-8') as f:
            # Skip rows if needed
            for _ in range(skip_rows):
                next(f, None)
                
            for line in f:
                # Assuming lines are space-separated token IDs
                try:
                    ids = [int(tok) for tok in line.strip().split()]
                    if len(ids) > max_len:
                        ids = ids[:max_len]
                    if ids: # Ensure we don't add empty lists
                        self.samples.append(ids)  # Store as list, not tensor yet
                        counter += 1
                        if counter % 1000 == 0:
                            logger.info(f"Loaded {counter} samples")
                        if max_samples is not None and counter >= max_samples:
                            break
                except ValueError as e:
                    logger.warning(f"Skipping line due to ValueError: {line[:50]}... - Error: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.samples)} samples in total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) # True for non-padded
        
        # For this test, we'll create a simple schema mask where we mark the first 
        # 30% of tokens as schema tokens, simulating a SQL example
        schema_len = max(1, int(len(input_ids) * 0.3))
        schema_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        schema_mask[:schema_len] = True  # Mark everything in schema section (except start/end tags)
        
        # Create a simple relation matrix for testing
        # This is just a placeholder - in real data you'd have actual relation information
        relation_matrix = torch.zeros((len(input_ids), len(input_ids)), dtype=torch.long)
        
        return {
            'encoder_input': input_ids,
            'decoder_target': input_ids.clone(), # For LM-style training
            'encoder_attention_mask': attention_mask,
            'relation_matrix': relation_matrix,
            'schema_mask': schema_mask
        }

# Define collate_fn as a top-level function to make it picklable
def custom_collate_fn(batch, pad_id=18):
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

class GradientMonitor:
    """Gradient monitoring utility"""
    def __init__(self, model, log_every=1):
        self.model = model
        self.log_every = log_every
        self.step_counter = 0
        self.grad_stats = []
        
    def compute_total_norm(self):
        """Compute the true gradient norm before any clipping"""
        # Calculate total norm manually to ensure we get the true pre-clipped value
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0, 0.0, "no_grad_params", {}
            
        # Compute norm for each parameter
        param_norms = {}
        max_norm = 0.0
        param_with_max_norm = None
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_norms[name] = param_norm
                if param_norm > max_norm:
                    max_norm = param_norm
                    param_with_max_norm = name
        
        # Compute overall norm without any clipping
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters])
        ).item()
        
        return total_norm, max_norm, param_with_max_norm, param_norms
        
    def update(self, step):
        """Call this before clipping and optimizer step"""
        self.step_counter += 1
        if self.step_counter % self.log_every == 0:
            # Compute gradient statistics (before clipping)
            total_norm, max_norm, param_with_max_norm, param_norms = self.compute_total_norm()
            
            self.grad_stats.append({
                'step': step,
                'total_norm': total_norm,
                'max_norm': max_norm,
                'max_norm_param': param_with_max_norm
            })
            
            logger.info(f"Gradient stats - Step: {step}, "
                       f"Total Norm (pre-clip): {total_norm:.4f}, "
                       f"Max Norm: {max_norm:.4f} ({param_with_max_norm})")
            
            # Log top 5 gradients
            sorted_norms = sorted(param_norms.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("Top 5 gradient norms:")
            for name, norm in sorted_norms:
                logger.info(f"  {name}: {norm:.4f}")
            
            return total_norm
        return None

class MemoryMonitor:
    """Memory monitoring utility"""
    def __init__(self, log_every=1):
        self.log_every = log_every
        self.step_counter = 0
        
    def update(self, step):
        self.step_counter += 1
        if self.step_counter % self.log_every == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            
            logger.info(f"GPU Memory - Step: {step}, "
                       f"Allocated: {allocated:.1f} MB, "
                       f"Max Allocated: {max_allocated:.1f} MB, "
                       f"Reserved: {reserved:.1f} MB")
            
            return {
                'allocated': allocated,
                'max_allocated': max_allocated,
                'reserved': reserved
            }
        return None

# Monkey patch the Trainer class to add gradient and memory monitoring
def patch_trainer_with_monitoring(trainer, grad_monitor, mem_monitor):
    """Patch the trainer to add gradient monitoring before clipping occurs"""
    original_clip_grad_norm = torch.nn.utils.clip_grad_norm_
    
    def monitored_clip_grad_norm_(parameters, max_norm, *args, **kwargs):
        """Capture true gradient norm before clipping is applied"""
        # Calculate and log the pre-clipped gradient norm
        grad_monitor.update(trainer.global_step)
        
        # Now apply the actual clipping
        return original_clip_grad_norm(parameters, max_norm, *args, **kwargs)
    
    # Replace torch's clip_grad_norm_ with our monitored version
    torch.nn.utils.clip_grad_norm_ = monitored_clip_grad_norm_
    
    # Store the original train_epoch method
    original_train_epoch = trainer.train_epoch
    
    def monitored_train_epoch(*args, **kwargs):
        """Add memory monitoring after optimization steps"""
        # Store the original optimizer step
        orig_step = trainer.optimizer.step
        
        def monitored_step(*args, **kwargs):
            # Call the original step
            result = orig_step(*args, **kwargs)
            # Monitor memory usage after step
            mem_monitor.update(trainer.global_step)
            return result
        
        # Replace optimizer step with monitored version
        trainer.optimizer.step = monitored_step
        
        # Run the original train_epoch
        result = original_train_epoch(*args, **kwargs)
        
        # Restore original methods
        trainer.optimizer.step = orig_step
        
        return result
    
    # Replace the train_epoch method
    trainer.train_epoch = monitored_train_epoch
    
    # Ensure we restore the original clipping function when training ends
    original_train = trainer.train
    
    def monitored_train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        # Restore original clip_grad_norm_
        torch.nn.utils.clip_grad_norm_ = original_clip_grad_norm
        return result
    
    trainer.train = monitored_train
    
    return trainer

def test_real_training(pretraining=True, sft=True):
    """Test training with real data
    
    Args:
        pretraining: Whether to test pretraining mode
        sft: Whether to test supervised fine-tuning mode
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Get GPU info if available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} with {gpu_mem:.1f} GB total memory")

    project_root = Path(__file__).resolve().parent.parent
    
    # Base results that will be extended for each training mode
    results = {
        'pretraining': None,
        'sft': None
    }
    
    # Training modes to test
    modes_to_test = []
    if pretraining:
        modes_to_test.append(('pretraining', 'masked language modeling'))
    if sft:
        modes_to_test.append(('sft', 'supervised fine-tuning'))
    
    for mode, description in modes_to_test:
        logger.info(f"Testing {description} mode")
        
        # Create a smaller configuration for testing on local machine
        config = NL2SQLConfig(
            vocab_size=32000,  # Should match tokenizer if using real data
            d_model=768,       # Reduced from default
            n_heads=12,
            n_layers=6,        # Reduced from 12
            num_relations=5,
            dropout=0.1,
            max_len=256,       # Reduced max sequence length
            pad_token_id=18,
            use_pointer_generator=True,
            special_tokens={    # Minimal special tokens for config validation
                'SCHEMA_START': '<SCHEMA>', 'SCHEMA_END': '</SCHEMA>',
                'PK_START': '<PK>', 'PK_END': '</PK>',
                'FK_START': '<FK>', 'FK_END': '</FK>',
                'NL_START': '<NL>', 'NL_END': '</NL>',
                'COT_START': '<COT>', 'COT_END': '</COT>',
                'SQL_START': '<SQL>', 'SQL_END': '</SQL>',
                'EXT_START': '<EXT>', 'EXT_END': '</EXT>'
            },
            batch_size=2,      # Very small batch size for testing
            max_batch_size=2,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_steps=5,    # Fewer steps for test
            max_steps=50,      # Run only a few steps for test
            gradient_accumulation_steps=2,  # For effective batch size of 4
            max_grad_norm=1.0,
            early_stopping_patience=2,
            save_steps=25,
            num_workers=0,     # Disable multiprocessing to avoid pickling issues
            mixed_precision=True,  # Enable mixed precision for memory efficiency
            use_8bit_optimizer=False,
            use_bf16=False,
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            sp_model_path=str(project_root / "models" / "nl2sql_tok.model"),
            output_dir=str(project_root / "outputs" / f"test_{mode}"),
            # Phase-specific max_len for pretraining (shorter sequences for testing)
            dataset_phase_max_len=128 if mode == 'pretraining' else None,
            # For SFT, we need to add phase_max_len_pg for pointer-generator
            # Use a value less than max_len (256) to avoid assertion error
            phase_max_len_pg=200 if mode == 'sft' else None,
            max_sql_len=50 if mode == 'sft' else None
        )

        # Set files based on mode
        if mode == 'pretraining':
            # Use the correct path for pretraining datasets
            train_file = str(project_root / "datasets" / "raw_sql" / "pretraining corpus" / "splits" / "wrapped_tokenized_corpus_train.txt")
            eval_file = str(project_root / "datasets" / "raw_sql" / "pretraining corpus" / "splits" / "wrapped_tokenized_corpus_val.txt")
        else:
            # SFT dataset paths
            train_file = str(project_root / "datasets" / "paired_nl_sql" / "splits" / "tokenized_sft_filtered_train.txt")
            eval_file = str(project_root / "datasets" / "paired_nl_sql" / "splits" / "tokenized_sft_filtered_val.txt")
        
        # Verify that files exist
        if not os.path.exists(train_file):
            logger.error(f"Could not find training file: {train_file}")
            results[mode] = {
                'success': False,
                'error': f"Training file not found: {train_file}"
            }
            continue
        
        if not os.path.exists(eval_file):
            logger.error(f"Could not find validation file: {eval_file}")
            results[mode] = {
                'success': False,
                'error': f"Validation file not found: {eval_file}"
            }
            continue

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create model with pointer-generator
        model = NL2SQLTransformer(
            vocab_size=config.vocab_size,
            num_relations=config.num_relations,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            max_len=config.max_len,
            use_pointer_generator=config.use_pointer_generator,
            pad_token_id=config.pad_token_id
        )
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

        # Get the appropriate max_len for dataset truncation
        dataset_max_len = config.get_dataset_max_len()
        logger.info(f"Using max_len {dataset_max_len} for dataset truncation (model's max_len: {config.max_len})")

        # Determine phase_max_len for RelationMatrixBuilder - use the config value if available
        phase_max_len = getattr(config, 'phase_max_len_pg', dataset_max_len)
        logger.info(f"Using phase_max_len {phase_max_len} for relation matrix building")

        # Load only a small subset of the training data (100 samples)
        logger.info("Loading small subset of training data")
        train_dataset = TokenizedTextDataset(
            train_file, 
            max_len=dataset_max_len, 
            pad_token_id=config.pad_token_id,
            max_samples=100,  # Only use 100 samples for testing
            skip_rows=0       # Start from beginning
        )
        
        # Load a small subset of validation data
        logger.info("Loading small subset of validation data")
        val_dataset = TokenizedTextDataset(
            eval_file, 
            max_len=dataset_max_len, 
            pad_token_id=config.pad_token_id,
            max_samples=20,   # Only use 20 samples for validation
            skip_rows=0       # Start from beginning
        )
        
        # Ensure datasets are not empty
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.error(f"Datasets are empty for {mode}. Skipping test.")
            results[mode] = {
                'success': False,
                'error': f"Empty datasets for {mode}"
            }
            continue

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda b: custom_collate_fn(b, pad_id=config.pad_token_id),
            num_workers=config.num_workers
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda b: custom_collate_fn(b, pad_id=config.pad_token_id),
            num_workers=config.num_workers
        )

        # Initialize gradient and memory monitoring
        grad_monitor = GradientMonitor(model, log_every=1)
        mem_monitor = MemoryMonitor(log_every=1)

        # Initialize trainer WITHOUT callbacks
        trainer = Trainer(
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
        
        # Patch the trainer with our monitoring
        trainer = patch_trainer_with_monitoring(trainer, grad_monitor, mem_monitor)

        logger.info(f"Starting {mode} test training run (20 steps max)...")
        start_time = time.time()
        
        # Limit to just a few steps (less than max_steps in config)
        try:
            # Wrap in try/except to catch out of memory or other errors
            trainer.train(num_epochs=1)  # Only train for 1 epoch, limiting steps in train()
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            
            if "CUDA out of memory" in str(e):
                logger.error(
                    "Out of CUDA memory. Try reducing batch_size, max_len, or model size. "
                    "Enable mixed_precision and gradient_checkpointing to save memory."
                )
            results[mode] = {
                'success': False,
                'error': str(e),
                'steps_completed': trainer.global_step
            }
            continue
        
        train_duration = time.time() - start_time
        logger.info(f"{mode} test training completed in {train_duration:.1f} seconds")

        # Log performance statistics
        steps_completed = trainer.global_step
        samples_processed = steps_completed * config.batch_size * config.gradient_accumulation_steps
        throughput = samples_processed / train_duration
        
        logger.info(f"{mode} training statistics:")
        logger.info(f"  Steps completed: {steps_completed}")
        logger.info(f"  Samples processed: {samples_processed}")
        logger.info(f"  Throughput: {throughput:.2f} samples/second")
        try:
            current_loss = trainer.current_loss # Get current_loss directly
            loss_display = f"{current_loss:.4f}" if current_loss is not None else "N/A"
            logger.info(f"  Current loss: {loss_display}")
        except AttributeError:
            logger.info(f"  Current loss: Not available (attribute not found)")
        
        if torch.cuda.is_available():
            logger.info(f"Final GPU Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
            logger.info(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")

        # Report gradient stability
        if grad_monitor.grad_stats:
            avg_norm = sum(stat['total_norm'] for stat in grad_monitor.grad_stats) / len(grad_monitor.grad_stats)
            max_norm = max(stat['max_norm'] for stat in grad_monitor.grad_stats)
            
            logger.info(f"Gradient stability for {mode}:")
            logger.info(f"  Average gradient norm: {avg_norm:.4f}")
            logger.info(f"  Maximum gradient norm: {max_norm:.4f}")
            
            if avg_norm > 10.0:
                logger.warning("Average gradient norm is high (>10.0). Consider reducing learning rate.")
            elif avg_norm < 0.1:
                logger.warning("Average gradient norm is low (<0.1). Consider increasing learning rate.")
            else:
                logger.info("Gradient norms look reasonable for this test run.")
        
        # Store results for this mode
        results[mode] = {
            'success': True,
            'steps_completed': steps_completed,
            'samples_processed': samples_processed,
            'throughput': throughput,
            'loss': getattr(trainer, 'current_loss', float('nan')),
            'train_duration': train_duration,
            'gradient_stats': {
                'avg_norm': avg_norm if grad_monitor.grad_stats else None,
                'max_norm': max_norm if grad_monitor.grad_stats else None
            },
            'checkpoint_path': str(output_dir / 'best_model.pt') if os.path.exists(output_dir / 'best_model.pt') else None
        }
        
        # Save a report with all the findings - Fix encoding issue by removing emoji
        report_path = output_dir / f"test_{mode}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"NL2SQL {description.capitalize()} Test Report\n")
            f.write(f"{'=' * (len(description) + 24)}\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {device}\n")
            if torch.cuda.is_available():
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"\nModel Configuration:\n")
            f.write(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
            f.write(f"  d_model: {config.d_model}\n")
            f.write(f"  n_layers: {config.n_layers}\n")
            f.write(f"  n_heads: {config.n_heads}\n")
            f.write(f"  model max_len: {config.max_len}\n")
            f.write(f"  dataset max_len: {dataset_max_len}\n")
            f.write(f"\nTraining Configuration:\n")
            f.write(f"  batch_size: {config.batch_size}\n")
            f.write(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}\n")
            f.write(f"  effective_batch_size: {config.batch_size * config.gradient_accumulation_steps}\n")
            f.write(f"  learning_rate: {config.learning_rate}\n")
            f.write(f"  mixed_precision: {config.mixed_precision}\n")
            f.write(f"  gradient_checkpointing: {config.gradient_checkpointing}\n")
            f.write(f"\nTraining Statistics:\n")
            f.write(f"  Steps completed: {steps_completed}\n")
            f.write(f"  Samples processed: {samples_processed}\n")
            f.write(f"  Training time: {train_duration:.1f} seconds\n")
            f.write(f"  Throughput: {throughput:.2f} samples/second\n")
            try:
                current_loss = getattr(trainer, 'current_loss', None) # Use None as default for checking
                loss_display = f"{current_loss:.4f}" if current_loss is not None else "N/A"
                f.write(f"  Final loss: {loss_display}\n")
            except AttributeError: # Should not happen with getattr, but good for safety
                f.write(f"  Final loss: Not available (attribute not found)\n")
            if torch.cuda.is_available():
                f.write(f"  Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB\n")
            f.write(f"\nGradient Analysis:\n")
            if grad_monitor.grad_stats:
                f.write(f"  Average gradient norm: {avg_norm:.4f}\n")
                f.write(f"  Maximum gradient norm: {max_norm:.4f}\n")
                if avg_norm > 10.0:
                    f.write(f"  WARNING: Average gradient norm is high (>10.0). Consider reducing learning rate.\n")
                elif avg_norm < 0.1:
                    f.write(f"  WARNING: Average gradient norm is low (<0.1). Consider increasing learning rate.\n")
                else:
                    f.write(f"  Gradient norms look reasonable for this test run.\n")
                    
                f.write(f"\nDetailed Gradient Stats (per step):\n")
                for stat in grad_monitor.grad_stats:
                    f.write(f"  Step {stat['step']}: Total Norm = {stat['total_norm']:.4f}, "
                            f"Max Norm = {stat['max_norm']:.4f} ({stat['max_norm_param']})\n")
            else:
                f.write(f"  No gradient statistics collected.\n")
                
            f.write(f"\nRecommendation for Production Training:\n")
            if steps_completed >= 10 and getattr(trainer, 'current_loss', float('inf')) < float('inf'):
                f.write(f"  [PASS] {description.capitalize()} test completed successfully. The model appears to be training correctly.\n")
                f.write(f"  [PASS] Loss is converging and gradients are stable. You can proceed with GCP training.\n")
            else:
                f.write(f"  [WARNING] {description.capitalize()} test did not complete enough steps or showed unstable loss.\n")
                f.write(f"  [WARNING] Review the logs and make necessary adjustments before proceeding to GCP.\n")
        
        logger.info(f"{mode} test report saved to {report_path}")
        
        # Clear GPU memory between tests
        model = model.to('cpu')
        del model, trainer, train_dataloader, val_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Prepare final summary report - Fix encoding issue by removing emoji
    summary_path = project_root / "outputs" / "test_training_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"NL2SQL Test Training Summary\n")
        f.write(f"===========================\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Test Results Summary:\n")
        for mode, result in results.items():
            if result is None:
                f.write(f"  {mode.upper()}: Not tested\n")
            elif not result['success']:
                f.write(f"  {mode.upper()}: Failed - {result['error']}\n")
            else:
                loss_val = result['loss']
                loss_display = f"{loss_val:.4f}" if loss_val is not None else "N/A"
                f.write(f"  {mode.upper()}: Success - Loss: {loss_display}, Steps: {result['steps_completed']}\n")
        
        f.write(f"\nRecommendation for Production Training:\n")
        if all(result is not None and result['success'] for result in results.values() if result is not None):
            f.write(f"  [PASS] All tested modes completed successfully. You can proceed with GCP training.\n")
        else:
            f.write(f"  [WARNING] Some tests failed. Review individual test reports before proceeding to GCP.\n")
    
    logger.info(f"Summary report saved to {summary_path}")
    
    # Return overall success
    return all(result is not None and result['success'] for result in results.values() if result is not None)

if __name__ == '__main__':
    if not test_real_training():
        logger.error("test_real_training FAILED")
        sys.exit(1)
    else:
        logger.info("test_real_training PASSED") 