import os
import logging
import threading
import sentencepiece as spm
from typing import Dict, List, Optional, Union, Set

logger = logging.getLogger(__name__)

class NL2SQLTokenizer:
    """Wrapper for SentencePiece tokenizer with special token handling."""
    
    def __init__(self, sp_model_path: str, special_tokens: Dict[str, str]):
        """
        Initialize tokenizer.
        
        Args:
            sp_model_path: Path to SentencePiece model file
            special_tokens: Dictionary mapping special token names to their string values
        """
        if not os.path.exists(sp_model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {sp_model_path}")
            
        self.sp = spm.SentencePieceProcessor()
        if not self.sp.load(sp_model_path):
            raise ValueError(f"Failed to load SentencePiece model from {sp_model_path}")
            
        self.special_tokens = special_tokens
        self._lock = threading.Lock()  # Thread safety
        self.unk_id = self.sp.unk_id()
        self._validate_special_tokens()
        
        # Cache special token IDs
        self.special_token_ids = {
            name: self.sp.piece_to_id(piece)
            for name, piece in special_tokens.items()
        }
        
        # Validate uniqueness of special token IDs
        if len(set(self.special_token_ids.values())) != len(self.special_token_ids):
            raise ValueError("Duplicate special token IDs found")
        
    def _validate_special_tokens(self):
        """Validate that all special tokens exist in the vocabulary."""
        for name, piece in self.special_tokens.items():
            try:
                token_id = self.sp.piece_to_id(piece)
                if token_id == self.unk_id:
                    raise ValueError(f"Special token '{piece}' maps to unknown token")
            except KeyError:
                raise ValueError(f"Special token '{piece}' not found in vocabulary")
                
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        with self._lock:
            try:
                ids = self.sp.encode(text)
                if add_special_tokens:
                    return self._add_special_tokens(ids)
                return ids
            except Exception as e:
                logger.error(f"Failed to encode text: {e}")
                raise
                
    def _add_special_tokens(self, ids: List[int]) -> List[int]:
        """Add special tokens to the sequence."""
        # Add BOS and EOS tokens if they exist
        if 'BOS' in self.special_token_ids:
            ids.insert(0, self.special_token_ids['BOS'])
        if 'EOS' in self.special_token_ids:
            ids.append(self.special_token_ids['EOS'])
        return ids
            
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        with self._lock:
            try:
                if skip_special_tokens:
                    # Filter out special token IDs
                    ids = [id for id in ids if id not in self.special_token_ids.values()]
                return self.sp.decode(ids)
            except Exception as e:
                logger.error(f"Failed to decode tokens: {e}")
                raise
            
    def get_special_token_id(self, name: str) -> int:
        """Get ID for a special token by name."""
        if name not in self.special_token_ids:
            raise KeyError(f"Unknown special token: {name}")
        return self.special_token_ids[name]
        
    def get_special_token_name(self, token_id: int) -> Optional[str]:
        """Get name for a special token by ID."""
        for name, id in self.special_token_ids.items():
            if id == token_id:
                return name
        return None
        
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.get_piece_size()
        
    def save_pretrained(self, path: str):
        """Save tokenizer configuration."""
        os.makedirs(path, exist_ok=True)
        self.sp.save(f"{path}/sp.model")
        # Save special tokens mapping
        import json
        with open(f"{path}/special_tokens.json", 'w') as f:
            json.dump(self.special_tokens, f)
            
    @classmethod
    def from_pretrained(cls, path: str) -> 'NL2SQLTokenizer':
        """Load tokenizer from saved configuration."""
        import json
        with open(f"{path}/special_tokens.json", 'r') as f:
            special_tokens = json.load(f)
        return cls(f"{path}/sp.model", special_tokens) 