import os
import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from tokenizer import NL2SQLTokenizer
from config import NL2SQLConfig
from relation_matrix import RelationMatrixBuilder

logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning of NL2SQL model.
    Assumes data_file is a .txt file where each line contains space-separated
    integer token IDs representing a full concatenated sequence (e.g., schema-NL-COT-SQL).
    """
    
    _source_truncation_warning_logged = False
    
    def __init__(self, data_file: str, tokenizer: NL2SQLTokenizer, 
                 relation_builder: RelationMatrixBuilder, 
                 config: NL2SQLConfig):
        """
        Initialize the SFT dataset.
        
        Args:
            data_file (str): Path to the .txt data file (space-separated token IDs per line)
            tokenizer: Tokenizer instance
            relation_builder: RelationMatrixBuilder instance
            config (NL2SQLConfig): Configuration object
        """
        self.tokenizer = tokenizer
        self.relation_builder = relation_builder
        self.config = config
        self.max_len = config.get_dataset_max_len()
        self.pad_token_id = config.pad_token_id
        
        self.sql_start_token_id = self.tokenizer.get_special_token_id('SQL_START')
        
        logger.info(f"Loading SFT data from {data_file} (expected format: txt, space-separated token IDs per line)")
        self.examples: List[List[int]] = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    token_ids = [int(token_id_str) for token_id_str in line.split()]
                    self.examples.append(token_ids)
                except ValueError as e:
                    logger.error(f"Error parsing token IDs on line {line_num+1} in {data_file}: {e}. Line: '{line[:100]}...' Skipping.")
                    continue
        logger.info(f"Loaded {len(self.examples)} examples for SFT")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        full_token_ids = self.examples[idx]
        
        # --- Step 1: Split into encoder_input and target_tokens using SQL_START ---
        try:
            sql_start_index = full_token_ids.index(self.sql_start_token_id)
            # Encoder input is everything up to (but not including) SQL_START
            # This includes schema, NL, and COT if present.
            raw_encoder_input_tokens = full_token_ids[:sql_start_index]
            # Target (labels) is everything from SQL_START to the end
            raw_target_tokens = full_token_ids[sql_start_index:]
        except ValueError:
            logger.warning(f"SFTDataset: SQL_START token (ID: {self.sql_start_token_id}) not found in example {idx}. "
                         f"Example (first 100 tokens): {full_token_ids[:100]}. Using full sequence as encoder input and an empty target. This will likely fail or be ignored in training.")
            raw_encoder_input_tokens = full_token_ids
            raw_target_tokens = [] # Or handle error more gracefully, e.g. skip sample in collate_fn

        # --- Step 2: Potentially truncate the encoder input (schema+NL+COT part) for Pointer-Generator ---
        sft_pg_source_max_len = self.config.phase_max_len_pg # Max length for schema+NL+COT if PG is used

        if self.config.use_pointer_generator and sft_pg_source_max_len is not None and \
           len(raw_encoder_input_tokens) > sft_pg_source_max_len:
            if not SFTDataset._source_truncation_warning_logged:
                logger.warning(
                    f"SFT source (schema+NL+COT) length {len(raw_encoder_input_tokens)} exceeds phase_max_len_pg {sft_pg_source_max_len}. "
                    f"Truncating from the beginning of source part. This warning is shown once."
                )
                SFTDataset._source_truncation_warning_logged = True
            
            amount_to_cut = len(raw_encoder_input_tokens) - sft_pg_source_max_len
            final_encoder_input_tokens = raw_encoder_input_tokens[amount_to_cut:]
        else:
            final_encoder_input_tokens = raw_encoder_input_tokens
            
        # --- Step 3: Apply overall self.max_len truncation ---
        # self.max_len is from config.get_dataset_max_len() -> SFT's phase_max_len (e.g. 1664) or model.max_len
        
        # Truncate final_encoder_input_tokens if it exceeds self.max_len
        if len(final_encoder_input_tokens) > self.max_len:
            final_encoder_input_tokens = final_encoder_input_tokens[:self.max_len]
        
        # Truncate target_tokens if it exceeds self.max_len
        final_target_tokens = raw_target_tokens
        if len(final_target_tokens) > self.max_len:
            final_target_tokens = final_target_tokens[:self.max_len]
        
        # Create tensors
        input_ids = torch.tensor(final_encoder_input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool) # True for non-padded
        labels = torch.tensor(final_target_tokens, dtype=torch.long)
        
        # --- Step 4: Build relation matrix and schema mask for pointer-generator ---
        # These are built based on the final_encoder_input_tokens.
        
        schema_meta: List = [] # Type hint, from relation_matrix.py
        relation_matrix = torch.zeros((len(input_ids), len(input_ids)), dtype=torch.long) # Default
        schema_mask_for_pg = torch.zeros_like(input_ids, dtype=torch.bool) # Default

        if self.config.use_pointer_generator and len(input_ids) > 0 : # Only build if PG is on and input is not empty
            try:
                # Schema parsing is done on the final_encoder_input_tokens.
                schema_meta = self.relation_builder.parse_schema_tokens(final_encoder_input_tokens)
                
                if not schema_meta: 
                    # Since we don't have original schema_str, we can't check if it was non-empty.
                    # We log if schema_meta is empty and PG is active.
                    logger.debug(f"SFTDataset: No schema tokens parsed by RelationMatrixBuilder for example {idx} from encoder input. "
                                 f"Pointer-generator copy may be ineffective. Input (first 50 tokens): {final_encoder_input_tokens[:50]}")

                relation_matrix = self.relation_builder.build_relation_matrix(final_encoder_input_tokens, schema_meta)
                    
                for token_s in schema_meta: # token_s is a SchemaToken object
                    # Ensure spans are within the (potentially self.max_len truncated) input_ids
                    start_idx, end_idx = token_s.span_start, token_s.span_end
                    # Check if span is valid for current input_ids length
                    if start_idx < len(input_ids) and end_idx < len(input_ids):
                        schema_mask_for_pg[start_idx : end_idx + 1] = True
                    elif start_idx < len(input_ids): # Partial span at the end due to truncation
                        schema_mask_for_pg[start_idx : len(input_ids)] = True
                
                # No direct equivalent of the old fallback warning logic as we don't have schema_str
                # If schema_meta is empty and PG is on, schema_mask_for_pg will be all False, PG won't copy. This is safe.

            except Exception as e:
                logger.error(f"Error building relation matrix or schema mask for SFT example {idx}: {e}", exc_info=True)
                # Fallback to zero/false tensors (already initialized)
        
        return {
            'encoder_input': input_ids,
            'decoder_target': labels,
            'encoder_attention_mask': attention_mask, 
            'relation_matrix': relation_matrix,  
            'schema_mask': schema_mask_for_pg    
        }

    @staticmethod
    def collate_fn(batch, pad_id=18):
        """
        Collate function for the dataloader.
        """
        # Filter out items where input_ids might be empty if SQL_START was not found and resulted in empty target
        # or if labels are empty. Such samples might cause issues.
        # However, an empty target might be valid if SQL_START is the last token.
        # An empty input_ids is more problematic.
        # For now, let's assume __getitem__ produces non-empty input_ids and labels where possible.
        # If labels are empty but input_ids is not, it might be a valid edge case (e.g. target is just <SQL_START><SQL_END>)
        
        # Max length for encoder_input in this batch
        max_enc_len = 0
        if batch: # Ensure batch is not empty
            max_enc_len = max(len(item['encoder_input']) for item in batch if item['encoder_input'] is not None)
        
        # Max length for decoder_target in this batch
        max_dec_len = 0
        if batch:
            max_dec_len = max(len(item['decoder_target']) for item in batch if item['decoder_target'] is not None)

        batch_size = len(batch)
        
        # Initialize tensors with pad_id
        # Encoder inputs and attention masks use max_enc_len
        encoder_input = torch.full((batch_size, max_enc_len), pad_id, dtype=torch.long)
        encoder_attention_mask = torch.zeros(batch_size, max_enc_len, dtype=torch.bool)
        
        # Decoder targets use max_dec_len. Note: In T5-style models, decoder input and target are shifted.
        # Here, 'decoder_target' is the label. The model itself handles creating decoder_input_ids.
        decoder_target = torch.full((batch_size, max_dec_len), pad_id, dtype=torch.long)
        
        # Relation matrix and schema mask are based on encoder's length (max_enc_len)
        relation_matrix = torch.zeros(batch_size, max_enc_len, max_enc_len, dtype=torch.long)
        schema_mask = torch.zeros(batch_size, max_enc_len, dtype=torch.bool)
        
        for i, item in enumerate(batch):
            enc_len = len(item['encoder_input'])
            if enc_len > 0:
                encoder_input[i, :enc_len] = item['encoder_input']
                encoder_attention_mask[i, :enc_len] = item['encoder_attention_mask'] # Should be all True for actual tokens
                # relation_matrix and schema_mask are also tied to encoder_input's length
                if item['relation_matrix'].ndim == 2 and item['relation_matrix'].shape[0] == enc_len and item['relation_matrix'].shape[1] == enc_len:
                    relation_matrix[i, :enc_len, :enc_len] = item['relation_matrix']
                else: # Handle case where relation_matrix might be a default empty one from __getitem__
                    if enc_len > 0 and item['relation_matrix'].numel() == 0: # If it's an empty tensor from a failed build
                         pass # Keep zeros
                    elif enc_len > 0 : # Log if shape mismatch for non-empty
                         logger.warning(f"Collator: Relation matrix shape mismatch for item {i}. Expected ({enc_len},{enc_len}), got {item['relation_matrix'].shape}. Using zeros.")

                if item['schema_mask'].ndim == 1 and item['schema_mask'].shape[0] == enc_len:
                     schema_mask[i, :enc_len] = item['schema_mask']
                else: # Handle case where schema_mask might be default/empty
                    if enc_len > 0 and item['schema_mask'].numel() == 0:
                        pass # Keep False
                    elif enc_len > 0:
                        logger.warning(f"Collator: Schema mask shape mismatch for item {i}. Expected ({enc_len}), got {item['schema_mask'].shape}. Using False.")


            dec_len = len(item['decoder_target'])
            if dec_len > 0:
                decoder_target[i, :dec_len] = item['decoder_target']
        
        return {
            'encoder_input': encoder_input,
            'decoder_target': decoder_target,
            'encoder_attention_mask': encoder_attention_mask,
            'relation_matrix': relation_matrix,
            'schema_mask': schema_mask
        }
