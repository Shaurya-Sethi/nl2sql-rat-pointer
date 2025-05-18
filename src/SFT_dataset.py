import os
import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from tokenizer import NL2SQLTokenizer
from config import NL2SQLConfig
from relation_matrix import RelationMatrixBuilder
import json

logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning of NL2SQL model."""
    
    def __init__(self, data_file, tokenizer, relation_builder, max_len=512, pad_token_id=0):
        """
        Initialize the SFT dataset.
        
        Args:
            data_file (str): Path to the data file
            tokenizer: Tokenizer instance
            relation_builder: RelationMatrixBuilder instance
            max_len (int): Maximum sequence length
            pad_token_id (int): Pad token ID
        """
        self.tokenizer = tokenizer
        self.relation_builder = relation_builder
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # Load data
        logger.info(f"Loading data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract components
        schema = example['schema']
        question = example['question']
        cot = example.get('cot', '')  # Chain of thought is optional
        sql = example['sql']
        
        # Construct input sequence
        input_seq = f"{schema}\n{question}"
        if cot:
            input_seq += f"\n{cot}"
        
        # Tokenize input and target
        input_tokens = self.tokenizer.encode(input_seq)
        target_tokens = self.tokenizer.encode(sql)
        
        # Truncate if necessary
        if len(input_tokens) > self.max_len:
            input_tokens = input_tokens[:self.max_len]
        if len(target_tokens) > self.max_len:
            target_tokens = target_tokens[:self.max_len]
        
        # Create tensors
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.tensor(target_tokens, dtype=torch.long)
        
        # Build relation matrix - use build_relation_matrix directly with tokenized input
        try:
            # Parse schema tokens from input tokens
            schema_tokens = self.relation_builder.parse_schema_tokens(input_tokens)
            
            # Validate schema tokens were found
            if not schema_tokens:
                logger.warning(f"No schema tokens found in example {idx}. Schema might be malformed.")
                # Create fallback relation matrix with all zeros
                relation_matrix = torch.zeros((len(input_tokens), len(input_tokens)), dtype=torch.long)
            else:
                # Build relation matrix using the parsed schema tokens
                relation_matrix = self.relation_builder.build_relation_matrix(input_tokens, schema_tokens)
                
            # Create schema mask for pointer-generator
            schema_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            
            # Set mask to True for all schema tokens (columns, tables, PKs, FKs)
            for token in schema_tokens:
                # Include span from start to end inclusive
                schema_mask[token.span_start:token.span_end+1] = True
                
            # Validate schema mask has at least one True value if schema tokens were found
            if len(schema_tokens) > 0 and not schema_mask.any():
                logger.warning(f"Schema mask creation failed for example {idx}. Using fallback.")
                # Create a simple fallback schema mask marking the first part of the input as schema
                # This is just a guess based on the typical structure
                schema_end_idx = min(len(input_tokens) // 3, 100)  # Estimate schema region
                schema_mask[:schema_end_idx] = True
        except Exception as e:
            logger.error(f"Error building relation matrix or schema mask for example {idx}: {e}")
            relation_matrix = torch.zeros((len(input_tokens), len(input_tokens)), dtype=torch.long)
            schema_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            # Create a simple fallback schema mask
            schema_end_idx = min(len(input_tokens) // 3, 100)  # Estimate schema region
            schema_mask[:schema_end_idx] = True
        
        return {
            'encoder_input': input_ids,
            'decoder_target': labels,
            'encoder_attention_mask': attention_mask,
            'relation_matrix': relation_matrix,
            'schema_mask': schema_mask  # For pointer-generator
        }

    @staticmethod
    def collate_fn(batch, pad_id=18):
        """
        Collate function for the dataloader.
        
        Args:
            batch: List of dictionaries containing encoder_input, decoder_target, encoder_attention_mask, relation_matrix, and schema_mask
            pad_id (int): Pad token ID
            
        Returns:
            Dictionary of batched tensors
        """
        max_len = max(len(item['encoder_input']) for item in batch)
        batch_size = len(batch)
        
        # Initialize tensors with pad_id
        encoder_input = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        decoder_target = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        encoder_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        relation_matrix = torch.zeros(batch_size, max_len, max_len, dtype=torch.long)
        schema_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)  # Initialize with False
        
        # Fill tensors
        for i, item in enumerate(batch):
            length = len(item['encoder_input'])
            encoder_input[i, :length] = item['encoder_input']
            decoder_target[i, :length] = item['decoder_target']
            encoder_attention_mask[i, :length] = item['encoder_attention_mask']
            relation_matrix[i, :length, :length] = item['relation_matrix']
            schema_mask[i, :length] = item['schema_mask']  # Copy schema mask with padding
        
        return {
            'encoder_input': encoder_input,
            'decoder_target': decoder_target,
            'encoder_attention_mask': encoder_attention_mask,
            'relation_matrix': relation_matrix,
            'schema_mask': schema_mask  # Include schema mask in batch
        }
