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
    
    _source_truncation_warning_logged = False
    
    def __init__(self, data_file: str, tokenizer: NL2SQLTokenizer, 
                 relation_builder: RelationMatrixBuilder, 
                 config: NL2SQLConfig,
                 pad_token_id: int = 0):
        """
        Initialize the SFT dataset.
        
        Args:
            data_file (str): Path to the data file
            tokenizer: Tokenizer instance
            relation_builder: RelationMatrixBuilder instance
            config (NL2SQLConfig): Configuration object
            pad_token_id (int): Pad token ID (will be taken from config)
        """
        self.tokenizer = tokenizer
        self.relation_builder = relation_builder
        self.config = config
        self.max_len = config.get_dataset_max_len()
        self.pad_token_id = config.pad_token_id
        
        # Load data
        logger.info(f"Loading SFT data from {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.examples = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(self.examples)} examples for SFT")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract components
        schema_str = example['schema']
        question_str = example['question']
        cot_str = example.get('cot', '')
        sql_str = example['sql']

        # --- Step 1: Prepare and potentially truncate the schema+NL part ---
        schema_nl_combined_str = f"{schema_str}\n{question_str}"
        schema_nl_tokens = self.tokenizer.encode(schema_nl_combined_str)

        sft_pg_source_max_len = self.config.phase_max_len_pg

        if self.config.use_pointer_generator and sft_pg_source_max_len is not None and \
           len(schema_nl_tokens) > sft_pg_source_max_len:
            if not SFTDataset._source_truncation_warning_logged:
                logger.warning(
                    f"SFT source (schema+NL) length {len(schema_nl_tokens)} exceeds phase_max_len_pg {sft_pg_source_max_len}. "
                    f"Truncating from the beginning of schema+NL part. This warning is shown once."
                )
                SFTDataset._source_truncation_warning_logged = True
            
            amount_to_cut = len(schema_nl_tokens) - sft_pg_source_max_len
            final_schema_nl_tokens = schema_nl_tokens[amount_to_cut:]
        else:
            final_schema_nl_tokens = schema_nl_tokens
            
        # --- Step 2: Construct the full encoder input sequence (tokens) ---
        # This uses the (potentially truncated by phase_max_len_pg) schema+NL tokens
        encoder_input_tokens = list(final_schema_nl_tokens) # Start with schema+NL

        if cot_str: # Append COT if present
            # Tokenize COT part with its preceding newline, as original SFTDataset seemed to imply
            cot_tokens = self.tokenizer.encode(f"\n{cot_str}")
            encoder_input_tokens.extend(cot_tokens)
        
        # --- Step 3: Tokenize SQL target ---
        target_tokens = self.tokenizer.encode(sql_str)
        
        # --- Step 4: Apply overall max_len truncation (e.g., 2048) ---
        # This self.max_len is from config.get_dataset_max_len() -> model.max_len for SFT
        if len(encoder_input_tokens) > self.max_len:
            encoder_input_tokens = encoder_input_tokens[:self.max_len]
        
        if len(target_tokens) > self.max_len: # Should be rare due to filtering
            target_tokens = target_tokens[:self.max_len]
        
        # Create tensors
        input_ids = torch.tensor(encoder_input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) # True for non-padded
        labels = torch.tensor(target_tokens, dtype=torch.long)
        
        # --- Step 5: Build relation matrix and schema mask for pointer-generator ---
        # These are built based on the final (potentially truncated by overall self.max_len) encoder_input_tokens.
        # parse_schema_tokens will operate on encoder_input_tokens. If schema_nl_combined_str was truncated
        # from the start, SCHEMA_START might be gone. parse_schema_tokens handles this (returns empty list).
        
        try:
            # Schema parsing is done on the full encoder_input_tokens.
            # Spans in schema_meta will be relative to the start of encoder_input_tokens.
            schema_meta = self.relation_builder.parse_schema_tokens(encoder_input_tokens)
            
            if not schema_meta and self.config.use_pointer_generator : # Log if schema is expected but not found
                 # Check if schema_str was non-empty to avoid warning for empty schema examples
                if schema_str.strip(): # Only warn if schema was supposed to be there
                    logger.debug(f"SFTDataset: No schema tokens parsed by RelationMatrixBuilder for example {idx}. "
                                 f"This might be due to schema+NL truncation or malformed schema. "
                                 f"Pointer-generator copy will be ineffective for this sample.")

            relation_matrix = self.relation_builder.build_relation_matrix(encoder_input_tokens, schema_meta)
                
            schema_mask_for_pg = torch.zeros_like(input_ids, dtype=torch.bool)
            if self.config.use_pointer_generator:
                for token_s in schema_meta:
                    # Ensure spans are within the (potentially self.max_len truncated) input_ids
                    start_idx, end_idx = token_s.span_start, token_s.span_end
                    if end_idx < len(input_ids):
                        schema_mask_for_pg[start_idx : end_idx + 1] = True
                    elif start_idx < len(input_ids): # Partial span at the end
                        schema_mask_for_pg[start_idx : len(input_ids)] = True
            
            # Fallback schema mask (from original code) for robustness if PG is on but schema parsing failed
            if self.config.use_pointer_generator and len(schema_meta) == 0 and schema_str.strip() and not schema_mask_for_pg.any():
                logger.warning(f"SFTDataset: Schema mask is all False for example {idx} despite non-empty schema_str and PG enabled. Applying fallback schema mask.")
                # This fallback is a guess, original code had a more complex one.
                # For PG, it's critical that schema_mask correctly identifies parts of input_ids.
                # If parse_schema_tokens is robust, this fallback should ideally not be hit often.
                # A simple fallback: mark first 1/3 of schema_nl_part if it was non-empty.
                # Length of the original schema_nl_tokens (before COT and overall truncation)
                # This is complex because final_schema_nl_tokens length is not directly available here
                # and input_ids might be truncated.
                # For now, rely on parse_schema_tokens. If it fails, schema_mask_for_pg remains False for copy.
                # The original SFTDataset fallback used input_tokens // 3, which is on the full encoder input.
                # This might not be what we want if phase_max_len_pg is much shorter.
                # The crucial thing is that schema_mask aligns with encoder_input_ids for the PG decoder.
                # If schema_meta is empty, schema_mask_for_pg will be all False, so PG won't copy. This is safe.
                pass # Schema mask will be all false, PG won't copy.

        except Exception as e:
            logger.error(f"Error building relation matrix or schema mask for SFT example {idx}: {e}", exc_info=True)
            # Fallback to zero/false tensors
            relation_matrix = torch.zeros((len(input_ids), len(input_ids)), dtype=torch.long)
            schema_mask_for_pg = torch.zeros_like(input_ids, dtype=torch.bool)
        
        return {
            'encoder_input': input_ids,
            'decoder_target': labels,
            'encoder_attention_mask': attention_mask, # This is for padding, not causal. Encoder handles its own.
            'relation_matrix': relation_matrix,   # For RelationAwareEncoder
            'schema_mask': schema_mask_for_pg     # For PointerGeneratorDecoder
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
