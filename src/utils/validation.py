import logging
import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from config import NL2SQLConfig

logger = logging.getLogger(__name__)

def validate_batch(batch: Dict[str, torch.Tensor], config: NL2SQLConfig) -> bool:
    """
    Validate batch shapes and types.
    
    Args:
        batch: Dictionary of tensors
        config: Model configuration
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check required keys
        required_keys = {'encoder_input', 'decoder_target', 'relation_matrix', 'encoder_attention_mask'}
        missing_keys = required_keys - set(batch.keys())
        if missing_keys:
            logger.error(f"Missing required keys in batch: {missing_keys}")
            return False

        batch_size = batch['encoder_input'].size(0)
        enc_seq_len = batch['encoder_input'].size(1)
        dec_seq_len = batch['decoder_target'].size(1)

        # Encoder-side shapes
        expected_shapes = {
            'encoder_input': (batch_size, enc_seq_len),
            'relation_matrix': (batch_size, enc_seq_len, enc_seq_len),
            'encoder_attention_mask': (batch_size, enc_seq_len)
        }
        for key, expected_shape in expected_shapes.items():
            if batch[key].shape != expected_shape:
                logger.error(f"Invalid shape for {key}: got {batch[key].shape}, expected {expected_shape}")
                return False

        # Decoder target can have any length >=2 (start + one target token minimum)
        if batch['decoder_target'].shape[0] != batch_size or dec_seq_len < 2:
            logger.error(f"Invalid shape for decoder_target: got {batch['decoder_target'].shape}, expected (batch_size, >=2)")
            return False

        # Dtype checks (unchanged)
        expected_dtypes = {
            'encoder_input': torch.long,
            'decoder_target': torch.long,
            'relation_matrix': torch.long,
            'encoder_attention_mask': torch.bool
        }
        for key, expected_dtype in expected_dtypes.items():
            if batch[key].dtype != expected_dtype:
                logger.error(f"Invalid dtype for {key}: got {batch[key].dtype}, expected {expected_dtype}")
                return False

        # Value checks (as before)
        if batch['encoder_input'].min() < 0 or batch['encoder_input'].max() >= config.vocab_size:
            logger.error(f"encoder_input values out of range [0, {config.vocab_size})")
            return False

        if batch['decoder_target'].min() < 0 or batch['decoder_target'].max() >= config.vocab_size:
            logger.error(f"decoder_target values out of range [0, {config.vocab_size})")
            return False

        if batch['relation_matrix'].min() < 0 or batch['relation_matrix'].max() >= config.num_relations:
            logger.error(f"relation_matrix values out of range [0, {config.num_relations})")
            return False

        # Validate attention mask (unchanged)
        if not batch['encoder_attention_mask'].any():
            logger.error("All positions masked in encoder_attention_mask")
            return False

        # Padding checks (unchanged)
        pad_mask = ~batch['encoder_attention_mask']
        if (batch['encoder_input'][pad_mask] != 18).any():
            logger.error("Non-pad tokens in padded positions")
            return False

        if (batch['relation_matrix'][pad_mask] != 0).any():
            logger.error("Non-zero relations in padded positions")
            return False

        # Sequence length/batch size checks for encoder_input only
        if enc_seq_len > config.max_len:
            logger.error(f"Sequence length {enc_seq_len} exceeds maximum {config.max_len}")
            return False

        if batch_size > config.max_batch_size:
            logger.error(f"Batch size {batch_size} exceeds maximum {config.max_batch_size}")
            return False

        # No check for decoder_target seq_len matching encoder_input seq_len!

        return True

    except Exception as e:
        logger.error(f"Error during batch validation: {e}")
        return False

def validate_model_inputs(
    encoder_input: torch.Tensor,
    decoder_input: torch.Tensor,
    relation_matrix: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    config: Optional[NL2SQLConfig] = None
) -> bool:
    """
    Validate model input tensors.
    
    Args:
        encoder_input: Encoder input tensor
        decoder_input: Decoder input tensor
        relation_matrix: Relation matrix tensor
        attention_mask: Optional attention mask
        config: Optional model configuration
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check shapes
        batch_size, seq_len = encoder_input.shape
        if decoder_input.shape != (batch_size, seq_len):
            logger.error("decoder_input shape mismatch")
            return False
            
        if relation_matrix.shape != (batch_size, seq_len, seq_len):
            logger.error("relation_matrix shape mismatch")
            return False
            
        if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
            logger.error("attention_mask shape mismatch")
            return False
            
        # Check dtypes
        if encoder_input.dtype != torch.long:
            logger.error("encoder_input must be long dtype")
            return False
            
        if decoder_input.dtype != torch.long:
            logger.error("decoder_input must be long dtype")
            return False
            
        if relation_matrix.dtype != torch.long:
            logger.error("relation_matrix must be long dtype")
            return False
            
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            logger.error("attention_mask must be bool dtype")
            return False
            
        # Check values if config provided
        if config is not None:
            if encoder_input.min() < 0 or encoder_input.max() >= config.vocab_size:
                logger.error(f"encoder_input values out of range [0, {config.vocab_size})")
                return False
                
            if decoder_input.min() < 0 or decoder_input.max() >= config.vocab_size:
                logger.error(f"decoder_input values out of range [0, {config.vocab_size})")
                return False
                
            if relation_matrix.min() < 0 or relation_matrix.max() >= config.num_relations:
                logger.error(f"relation_matrix values out of range [0, {config.num_relations})")
                return False
                
            # Check sequence lengths
            if seq_len > config.max_len:
                logger.error(f"Sequence length {seq_len} exceeds maximum {config.max_len}")
                return False
                
            # Check batch size
            if batch_size > config.max_batch_size:
                logger.error(f"Batch size {batch_size} exceeds maximum {config.max_batch_size}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error during input validation: {e}")
        return False

def validate_model_outputs(
    logits: torch.Tensor,
    encoder_output: torch.Tensor,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int
) -> bool:
    """
    Validate model output tensors.
    
    Args:
        logits: Logits tensor
        encoder_output: Encoder output tensor
        batch_size: Expected batch size
        seq_len: Expected sequence length
        vocab_size: Expected vocabulary size
        d_model: Expected model dimension
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check shapes
        if logits.shape != (batch_size, seq_len, vocab_size):
            logger.error(f"logits shape mismatch: got {logits.shape}, expected ({batch_size}, {seq_len}, {vocab_size})")
            return False
            
        if encoder_output.shape != (batch_size, seq_len, d_model):
            logger.error(f"encoder_output shape mismatch: got {encoder_output.shape}, expected ({batch_size}, {seq_len}, {d_model})")
            return False
            
        # Check for NaN/Inf values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error("NaN/Inf values in logits")
            return False
            
        if torch.isnan(encoder_output).any() or torch.isinf(encoder_output).any():
            logger.error("NaN/Inf values in encoder_output")
            return False
            
        # Check for extreme values
        if torch.abs(logits).max() > 100:
            logger.warning("Extreme values in logits")
            
        if torch.abs(encoder_output).max() > 100:
            logger.warning("Extreme values in encoder_output")
            
        return True
        
    except Exception as e:
        logger.error(f"Error during output validation: {e}")
        return False

def validate_metrics(metrics: Dict[str, float]) -> bool:
    """
    Validate training/validation metrics.
    
    Args:
        metrics: Dictionary of metric values
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Check for NaN/Inf values
        for name, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                logger.error(f"NaN/Inf value in metric {name}")
                return False
                
        # Check for extreme values
        for name, value in metrics.items():
            if abs(value) > 1e6:
                logger.warning(f"Extreme value in metric {name}: {value}")
                
        return True
        
    except Exception as e:
        logger.error(f"Error during metrics validation: {e}")
        return False