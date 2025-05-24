import os
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import yaml
import torch

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class NL2SQLConfig:
    """
    Configuration for NL2SQL model and training.
    
    Notes on mixed precision training:
    - When using mixed precision (mixed_precision=True), be aware that this causes
      some operations to use float16/bfloat16, which may lead to numeric instability.
    - The implementation has safeguards to ensure tensors have correct types (long for IDs, bool for masks),
      but custom code might need additional type checking.
    - For large batch sizes or deep models, consider enabling gradient_checkpointing to save memory.
    
    Notes on schema-aware training:
    - For pointer-generator to work correctly, schema_mask must be properly generated.
    - The schema parser in relation_matrix.py expects a specific format for schema tokens.
    - If schema parsing fails, the system will fall back to heuristic approaches, but quality may degrade.
    """
    # Model architecture
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    num_relations: int
    dropout: float
    max_len: int
    use_pointer_generator: bool  # Whether to use pointer-generator for copying schema tokens
    
    # Tokenizer
    special_tokens: Dict[str, str]
    
    # Training (will be set based on phase)
    batch_size: int
    max_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    early_stopping_patience: Optional[int]
    save_steps: int
    num_workers: int
    mixed_precision: bool
    use_8bit_optimizer: bool
    use_bf16: bool
    gradient_checkpointing: bool
    
    # Paths
    sp_model_path: str
    output_dir: str
    
    train_file: str
    eval_file: str
    
    # TensorBoard logging
    tensorboard_log_dir: Optional[str] = None  # Directory for TensorBoard logs, e.g., 'runs/exp1'
    log_every_n_steps: int = 10  # Log training metrics every N steps
    log_grad_norm: bool = True   # Whether to log gradient norms
    log_grad_histogram: bool = False  # Whether to log parameter histograms (more expensive)
    log_memory: bool = True  # Log GPU memory usage (if available)
    
    # Total number of epochs for training
    epochs: int
    
    # Parameters with default values
    pad_token_id: int = 18
    
    # Phase-specific parameters
    # This phase_max_len is for the current phase's dataset item length (e.g. pretrain can use 512)
    dataset_phase_max_len: Optional[int] = None 
    # This phase_max_len_pg is specifically for SFT pointer-generator's source (schema+NL) part
    phase_max_len_pg: Optional[int] = None 
    max_sql_len: Optional[int] = None    # Expected max length for SQL part in SFT (for clarity/logging)
    
    @classmethod
    def from_yaml(cls, yaml_path: str, phase: str = 'sft'):
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            phase: Training phase ('pretraining' or 'sft')
            
        Returns:
            NL2SQLConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If phase is invalid or required fields are missing
            KeyError: If required config sections are missing
        """
        # Validate phase
        if phase not in ['pretrain', 'sft']:
            raise ValueError(f"Invalid phase: {phase}. Must be 'pretrain' or 'sft'.")
            
        # Check file exists
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")
            
        # Validate config_dict has required sections
        required_sections = ['model', 'tokenizer', phase, 'paths', 'logging']
        missing_sections = [s for s in required_sections if s not in config_dict]
        if missing_sections:
            raise KeyError(f"Missing required config sections: {missing_sections}")
            
        # Get phase-specific config
        phase_config = config_dict[phase]
        
        # Check for required fields in phase config
        required_phase_fields = [
            'micro_batch_size', 'max_batch_size', 'learning_rate', 'weight_decay',
            'warmup_steps', 'max_steps', 'gradient_accumulation', 'max_grad_norm',
            'early_stopping_patience', 'save_steps', 'num_workers', 'mixed_precision',
            'use_8bit_optimizer', 'bf16', 'gradient_checkpointing', 'epochs'
        ]
        missing_fields = [f for f in required_phase_fields if f not in phase_config]
        if missing_fields:
            raise KeyError(f"Missing required fields in {phase} config: {missing_fields}")
            
        # Create config with model and tokenizer settings
        config = cls(
            # Model architecture
            vocab_size=config_dict['model']['vocab_size'],
            d_model=config_dict['model']['d_model'],
            n_heads=config_dict['model']['n_heads'],
            n_layers=config_dict['model']['n_layers'],
            num_relations=config_dict['model']['num_relations'],
            dropout=config_dict['model']['dropout'],
            max_len=config_dict['model']['max_len'],
            use_pointer_generator=config_dict['model'].get('use_pointer_generator', False),  # Default to False if not specified
            
            # Tokenizer
            special_tokens=config_dict['tokenizer']['special_tokens'],
            
            # Training (from phase-specific config)
            batch_size=phase_config['micro_batch_size'],
            max_batch_size=phase_config['max_batch_size'],
            learning_rate=phase_config['learning_rate'],
            weight_decay=phase_config['weight_decay'],
            warmup_steps=phase_config['warmup_steps'],
            max_steps=phase_config['max_steps'],
            gradient_accumulation_steps=phase_config['gradient_accumulation'],
            max_grad_norm=phase_config['max_grad_norm'],
            early_stopping_patience=phase_config['early_stopping_patience'],
            save_steps=phase_config['save_steps'],
            num_workers=phase_config['num_workers'],
            mixed_precision=phase_config['mixed_precision'],
            use_8bit_optimizer=phase_config['use_8bit_optimizer'],
            use_bf16=phase_config['bf16'],
            gradient_checkpointing=phase_config['gradient_checkpointing'],
            
            # Paths
            sp_model_path=config_dict['paths']['sp_model'],
            output_dir=config_dict['paths']['output_dir'],
            
            # TensorBoard logging
            tensorboard_log_dir=config_dict['logging'].get('tensorboard_log_dir'),
            log_every_n_steps=config_dict['logging'].get('log_every_n_steps', 10),
            log_grad_norm=config_dict['logging'].get('log_grad_norm', True),
            log_grad_histogram=config_dict['logging'].get('log_grad_histogram', False),
            log_memory=config_dict['logging'].get('log_memory', True),
            
            # Total Epochs
            epochs=phase_config['epochs'],
            
            # Phase-specific parameters
            # This phase_max_len is for the current phase's dataset item length (e.g. pretrain can use 512)
            dataset_phase_max_len=phase_config.get('max_len'), 
            # SFT specific lengths for PG and clarity
            phase_max_len_pg=phase_config.get('phase_max_len') if phase == 'sft' else None,
            max_sql_len=phase_config.get('max_sql_len') if phase == 'sft' else None,
            train_file=phase_config['train_file'],
            eval_file=phase_config['eval_file']
        )
        
        # Overwrite phase_max_len with the SFT-specific one if in SFT phase and it exists
        if phase == 'sft':
            sft_phase_max_len = phase_config.get('phase_max_len') # The new one specific for schema+NL
            if sft_phase_max_len is not None:
                config.phase_max_len_pg = sft_phase_max_len
        
        # For pointer-generator in SFT, ensure schema tokens are provided
        if phase == 'sft' and config.use_pointer_generator:
            required_schema_tokens = ['SCHEMA_START', 'SCHEMA_END', 'PK_START', 'PK_END', 'FK_START', 'FK_END']
            missing_tokens = [t for t in required_schema_tokens if t not in config.special_tokens]
            if missing_tokens:
                raise ValueError(
                    f"Pointer-generator requires schema tokens, but missing: {missing_tokens}. "
                    "Either provide these tokens or set use_pointer_generator=false."
                )
                
        # Special check for d_model and n_heads compatibility
        if config.d_model % config.n_heads != 0:
            raise ValueError(
                f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads}). "
                f"This error will cause dimension mismatch in multi-head attention."
            )
            
        # Check mixed precision settings
        if config.mixed_precision and config.use_bf16 and not torch_supports_bf16():
            logger.warning(
                "BF16 precision requested but your system may not support it. "
                "Training may fail. Consider setting bf16=false if errors occur."
            )
            
        # Ensure gradient_checkpointing and no_grad compatibility
        if config.gradient_checkpointing and not torch_supports_checkpoint():
            logger.warning(
                "Gradient checkpointing enabled but torch.utils.checkpoint may not be available. "
                "This could lead to errors during training."
            )
            
        # Validate and log for pointer-generator and phase_max_len
        if config.use_pointer_generator:
            if config.phase_max_len_pg is None:
                # If we are in a phase that should have it (e.g. SFT, but this check is general)
                # For now, this error is specific to SFT as per user request.
                # We determine current phase by checking if max_sql_len is set (proxy for SFT)
                is_sft_like_phase = config.max_sql_len is not None 
                if is_sft_like_phase: # Apply this stricter check for SFT-like phases
                    raise ValueError(
                        "Pointer-generator is enabled for SFT-like phase, but 'phase_max_len' is not set in config. "
                        "This is required for schema+NL truncation."
                    )
                else: # For other phases (like pretraining, if PG was used there), it's a warning
                    logger.warning(
                        "Using pointer-generator without phase_max_len specified. "
                        "For SFT, this will be an error. For other phases, memory efficiency might be suboptimal."
                    )
            else:
                logger.info(
                    f"Pointer-generator enabled (phase_max_len={config.phase_max_len_pg}, max_sql_len={config.max_sql_len})"
                )
            
        # Warn about very large batch sizes
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        if effective_batch > 128:
            logger.warning(
                f"Very large effective batch size: {effective_batch}. "
                "Consider gradient accumulation instead of large micro-batch size."
            )
            
        return config
    
    def __post_init__(self):
        """Validate configuration values."""
        # Model architecture validation
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.num_relations > 0, "num_relations must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert self.max_len > 0, "max_len must be positive"
        assert isinstance(self.use_pointer_generator, bool), "use_pointer_generator must be a boolean"
        
        # Model geometry validation
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        # Phase-specific max_len validation if it exists
        if self.dataset_phase_max_len is not None:
            assert self.dataset_phase_max_len > 0, "dataset_phase_max_len must be positive"
            assert self.dataset_phase_max_len <= self.max_len, "dataset_phase_max_len cannot exceed max_len"
        
        if self.phase_max_len_pg is not None: # For SFT PG schema+NL part
            assert self.phase_max_len_pg > 0, "phase_max_len_pg for SFT PG must be positive"
            assert self.phase_max_len_pg <= self.max_len, \
                f"phase_max_len_pg ({self.phase_max_len_pg}) cannot exceed model max_len ({self.max_len})"
            if self.dataset_phase_max_len is not None : # e.g. if SFT had its own overall max_len like pretrain
                 assert self.phase_max_len_pg <= self.dataset_phase_max_len, \
                    f"phase_max_len_pg ({self.phase_max_len_pg}) cannot exceed dataset_phase_max_len ({self.dataset_phase_max_len})"
        
        # Training validation
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.save_steps > 0, "save_steps must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.epochs > 0, "epochs must be positive"
        
        # Required special tokens
        required_tokens = [
            'SCHEMA_START', 'SCHEMA_END',
            'PK_START', 'PK_END',
            'FK_START', 'FK_END',
            'NL_START', 'NL_END',
            'COT_START', 'COT_END',
            'SQL_START', 'SQL_END',
            'EXT_START', 'EXT_END'
        ]
        for token in required_tokens:
            assert token in self.special_tokens, f"Missing required special token: {token}"
            
        # Validate file paths
        if not os.path.exists(self.sp_model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {self.sp_model_path}")
            
        assert self.pad_token_id == 18, "pad_token_id must be 18 to match tokenizer"
        
        # Additional validation warnings
        if self.mixed_precision and self.use_bf16:
            logger.info("Using mixed precision with BF16. Ensure your hardware supports BF16.")
            
        if self.max_len > 1024 and not self.gradient_checkpointing:
            logger.warning(
                f"Using large max_len ({self.max_len}) without gradient_checkpointing. "
                "This may lead to OOM errors with large models."
            )
            
        # Warn about very large batch sizes
        effective_batch = self.batch_size * self.gradient_accumulation_steps
        if effective_batch > 128:
            logger.warning(
                f"Very large effective batch size: {effective_batch}. "
                "Consider gradient accumulation instead of large micro-batch size."
            )
            
        # Validate and log for pointer-generator and phase_max_len
        if self.use_pointer_generator:
            if self.phase_max_len_pg is None:
                # If we are in a phase that should have it (e.g. SFT, but this check is general)
                # For now, this error is specific to SFT as per user request.
                # We determine current phase by checking if max_sql_len is set (proxy for SFT)
                is_sft_like_phase = self.max_sql_len is not None 
                if is_sft_like_phase: # Apply this stricter check for SFT-like phases
                    raise ValueError(
                        "Pointer-generator is enabled for SFT-like phase, but 'phase_max_len' is not set in config. "
                        "This is required for schema+NL truncation."
                    )
                else: # For other phases (like pretraining, if PG was used there), it's a warning
                    logger.warning(
                        "Using pointer-generator without phase_max_len specified. "
                        "For SFT, this will be an error. For other phases, memory efficiency might be suboptimal."
                    )
            else:
                logger.info(
                    f"Pointer-generator enabled (phase_max_len={self.phase_max_len_pg}, max_sql_len={self.max_sql_len})"
                )
            
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def get_dataset_max_len(self) -> int:
        """
        Get the appropriate max length for truncating full dataset items for the current phase.
        This uses dataset_phase_max_len if set (e.g. pretraining 512), otherwise model's global max_len.
        """
        return self.dataset_phase_max_len if self.dataset_phase_max_len is not None else self.max_len


def torch_supports_bf16() -> bool:
    """Check if torch supports bfloat16."""
    try:
        return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    except AttributeError: # Older torch versions
        try:
            return hasattr(torch, 'bfloat16')
        except ImportError: return False


def torch_supports_checkpoint() -> bool:
    """Check if torch supports checkpointing."""
    try:
        import torch.utils.checkpoint
        return True
    except ImportError:
        return False 