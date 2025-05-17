import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import yaml

@dataclass
class NL2SQLConfig:
    """Configuration for NL2SQL model and training."""
    # Model architecture
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    num_relations: int
    dropout: float
    max_len: int
    
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
    
    # Parameters with default values
    pad_token_id: int = 18
    
    @classmethod
    def from_yaml(cls, yaml_path: str, phase: str = 'sft'):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Get phase-specific config
        phase_config = config_dict[phase]
        
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
            output_dir=config_dict['paths']['output_dir']
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
        
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False) 