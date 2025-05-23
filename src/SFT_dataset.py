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
    
    _source_truncation_warning_logged = False # Static class variable for one-time warning
    
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
        self.max_len = config.get_dataset_max_len() # SFT's phase_max_len (e.g. 2048)
        self.pad_token_id = config.pad_token_id
        if self.pad_token_id is None:
            logger.warning("SFTDataset: config.pad_token_id is None. Defaulting to 18 for padding.")
            self.pad_token_id = 18 # Fallback as per user instruction context

        try:
            self.vocab_size = self.tokenizer.vocab_size
        except AttributeError:
            try:
                self.vocab_size = self.tokenizer.get_vocab_size()
            except AttributeError:
                logger.error("SFTDataset: Tokenizer has no 'vocab_size' or 'get_vocab_size()'. OOV checks will be skipped.")
                self.vocab_size = float('inf') # Effectively skips OOV check

        self.min_len = 4 # SFT rule: len < 4 is too short
        
        self.sql_start_token_id = self.tokenizer.get_special_token_id('SQL_START')
        self.sql_end_token_id = self.tokenizer.get_special_token_id('SQL_END')
        
        # For SFT truncation, we want to preserve SQL_END if it's the last token.
        # Based on user feedback, a generic 'EOS' token is not used to mark overall sequence end.
        # The sequences are concatenations like schema-NL-COT-SQL, ending with SQL_END.
        self.sft_end_special_tokens = {self.sql_end_token_id}
        logger.info(f"SFTDataset: Using {{ {self.sql_end_token_id}: '{self.tokenizer.special_tokens.get('SQL_END')}' }} as special end tokens for SFT truncation preservation.")

        self.dropped_count = 0
        self.truncated_count = 0
        self._summary_logged_first_time = False # To log summary only once initially
        self.last_logged_dropped_count = 0
        self.last_logged_truncated_count = 0

        logger.info(f"SFTDataset: Initializing with data_file='{data_file}', max_len={self.max_len}, min_len={self.min_len}, pad_id={self.pad_token_id}")
        self.examples: List[List[int]] = []
        raw_lines_count = 0
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                raw_lines_count += 1
                line = line.strip()
                if not line:
                    # This initial drop is for truly empty lines, not all-pad lines (handled in __getitem__)
                    logger.debug(f"SFTDataset: Skipping empty line {line_num+1} in {data_file} at load time.")
                    self.dropped_count += 1
                    continue
                try:
                    token_ids = [int(token_id_str) for token_id_str in line.split()]
                    if not token_ids: # Line had only whitespace, resulting in empty list
                        logger.debug(f"SFTDataset: Skipping line {line_num+1} (whitespace only) in {data_file} at load time.")
                        self.dropped_count += 1
                        continue
                    self.examples.append(token_ids)
                except ValueError as e:
                    logger.warning(f"SFTDataset: Error parsing token IDs (non-integer) on line {line_num+1} in {data_file}: {e}. Line: '{line[:100]}...' Skipping line at load time.")
                    self.dropped_count += 1
                    continue
        
        logger.info(f"SFTDataset: Loaded {len(self.examples)} examples from {raw_lines_count} raw lines. Initial drops (empty/parse error): {self.dropped_count}")
        self._log_counts_summary_if_needed(dataset_name="SFTDataset", force_log=True) # Log initial state

    def _log_counts_summary_if_needed(self, dataset_name: str, force_log: bool = False):
        current_dropped = self.dropped_count
        current_truncated = self.truncated_count
        
        has_significant_counts = (current_dropped > 0 or current_truncated > 0)
        counts_changed = (current_dropped != self.last_logged_dropped_count or 
                          current_truncated != self.last_logged_truncated_count)

        if has_significant_counts and (force_log or not self._summary_logged_first_time or counts_changed):
            logger.info(f"{dataset_name} Counts: Dropped={current_dropped}, Truncated={current_truncated}. (Total loaded: {len(self.examples) + self.dropped_count})")
            self.last_logged_dropped_count = current_dropped
            self.last_logged_truncated_count = current_truncated
            self._summary_logged_first_time = True

    def __len__(self):
        # Log summary when __len__ is called, as it might be before full iteration.
        self._log_counts_summary_if_needed(dataset_name="SFTDataset", force_log=True)
        return len(self.examples) # Number of potentially processable examples

    def __getitem__(self, idx):
        if idx >= len(self.examples):
             raise IndexError(f"Index {idx} out of range for SFTDataset with {len(self.examples)} examples.")
        original_token_ids = self.examples[idx]
        token_ids = list(original_token_ids) # Work with a copy
        
        # 1. Trim leading/trailing pad_ids
        start_idx = 0
        while start_idx < len(token_ids) and token_ids[start_idx] == self.pad_token_id:
            start_idx += 1
        end_idx = len(token_ids)
        while end_idx > start_idx and token_ids[end_idx - 1] == self.pad_token_id:
            end_idx -= 1
        token_ids = token_ids[start_idx:end_idx]

        # 2. Rule: Empty / All tokens equal to pad_id (already handled by trim + empty check)
        if not token_ids:
            logger.debug(f"SFTDataset [EX {idx}]: Sequence empty after trimming pad_id ({self.pad_token_id}). Dropping.")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None

        # 3. Rule: OOV ID (token < 0 or token >= vocab_size)
        if self.vocab_size != float('inf'): # Only if vocab_size is known
            for i, token_id in enumerate(token_ids):
                if not (0 <= token_id < self.vocab_size):
                    logger.debug(f"SFTDataset [EX {idx}]: OOV token ID {token_id} at pos {i} (vocab size {self.vocab_size}). Dropping. Sample: {token_ids[:10]}...")
                    self.dropped_count += 1
                    self._log_counts_summary_if_needed(dataset_name="SFTDataset")
                    return None
        
        # 4. Rule: Too short (SFT: len < 4)
        if len(token_ids) < self.min_len:
            logger.debug(f"SFTDataset [EX {idx}]: Sequence too short (len {len(token_ids)} < min_len {self.min_len}) after trimming. Dropping. Sample: {token_ids}")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None

        # 5. Rule: Too long (SFT > max_len (2048))
        if len(token_ids) > self.max_len:
            self.truncated_count += 1
            # self._log_counts_summary_if_needed(dataset_name="SFTDataset") # Logged in __len__ or by interval
            original_last_token = token_ids[-1]
            token_ids = token_ids[:self.max_len]
            if self.max_len > 0 and original_last_token in self.sft_end_special_tokens and token_ids[-1] != original_last_token:
                token_ids[-1] = original_last_token # Preserve special end token if possible
            logger.debug(f"SFTDataset [EX {idx}]: Sequence truncated from {len(original_token_ids)} to {len(token_ids)} (max_len {self.max_len}).")
        
        full_token_ids_processed = token_ids # This is now the validated and possibly truncated list of ints

        # --- Original SFTDataset logic starts, using full_token_ids_processed ---
        try:
            sql_start_index = full_token_ids_processed.index(self.sql_start_token_id)
            raw_encoder_input_tokens = full_token_ids_processed[:sql_start_index]
            raw_target_tokens = full_token_ids_processed[sql_start_index:]
        except ValueError:
            logger.warning(f"SFTDataset [EX {idx}]: SQL_START (ID {self.sql_start_token_id}) not found in validated sample. Len: {len(full_token_ids_processed)}. Sample: {full_token_ids_processed[:30]}... Using full as encoder input, empty target. May fail downstream.")
            raw_encoder_input_tokens = full_token_ids_processed
            raw_target_tokens = [] 

        sft_pg_source_max_len = self.config.phase_max_len_pg
        final_encoder_input_tokens = raw_encoder_input_tokens
        if self.config.use_pointer_generator and sft_pg_source_max_len is not None and \
           len(raw_encoder_input_tokens) > sft_pg_source_max_len:
            if not SFTDataset._source_truncation_warning_logged: # Static class variable
                logger.warning(
                    f"SFTDataset: PG source (schema+NL+COT) length {len(raw_encoder_input_tokens)} exceeds phase_max_len_pg {sft_pg_source_max_len}. "
                    f"Truncating from the beginning of source part. This warning is shown once per class instance."
                )
                SFTDataset._source_truncation_warning_logged = True # Mark per instance
            amount_to_cut = len(raw_encoder_input_tokens) - sft_pg_source_max_len
            final_encoder_input_tokens = raw_encoder_input_tokens[amount_to_cut:]
        
        if not final_encoder_input_tokens:
             logger.debug(f"SFTDataset [EX {idx}]: final_encoder_input_tokens became empty after PG truncation or SQL_START split. Dropping. Original: {original_token_ids[:30]}")
             self.dropped_count +=1
             self._log_counts_summary_if_needed(dataset_name="SFTDataset")
             return None
        
        # Create tensors
        input_ids = torch.tensor(final_encoder_input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.tensor(raw_target_tokens, dtype=torch.long)
        
        # Labels should not exceed overall max_len. Already handled by initial truncation of full_token_ids_processed.
        # If raw_target_tokens is empty (e.g. SQL_START was last or not found), labels tensor will be empty.
        # This is typically handled by loss functions (ignore_index) or collator (skip if all labels empty).

        schema_meta: List = []
        relation_matrix = torch.zeros((len(input_ids), len(input_ids)), dtype=torch.long)
        schema_mask_for_pg = torch.zeros_like(input_ids, dtype=torch.bool)

        if self.config.use_pointer_generator and len(input_ids) > 0:
            try:
                schema_meta = self.relation_builder.parse_schema_tokens(final_encoder_input_tokens)
                if not schema_meta:
                    logger.debug(f"SFTDataset [EX {idx}]: No schema tokens parsed for PG from encoder input. PG copy may be ineffective. Input: {final_encoder_input_tokens[:50]}")
                relation_matrix = self.relation_builder.build_relation_matrix(final_encoder_input_tokens, schema_meta)
                for token_s in schema_meta:
                    start_idx_s, end_idx_s = token_s.span_start, token_s.span_end
                    if start_idx_s < len(input_ids) and end_idx_s < len(input_ids):
                        schema_mask_for_pg[start_idx_s : end_idx_s + 1] = True
                    elif start_idx_s < len(input_ids):
                        schema_mask_for_pg[start_idx_s : len(input_ids)] = True
            except Exception as e:
                logger.error(f"SFTDataset [EX {idx}]: Error building relation matrix/schema mask: {e}", exc_info=True)
        
        return {
            'encoder_input': input_ids,
            'decoder_target': labels,
            'encoder_attention_mask': attention_mask, 
            'relation_matrix': relation_matrix,  
            'schema_mask': schema_mask_for_pg    
        }

    @staticmethod
    def collate_fn(batch, pad_id=18): # Default pad_id as per user context if not passed from DataLoader
        original_batch_len = len(batch)
        # 1. Filter out None items (samples dropped by __getitem__)
        batch = [item for item in batch if item is not None]
        
        if not batch:
            if original_batch_len > 0:
                 logger.warning("SFTDataset.collate_fn: Batch is empty after __getitem__ filtering. Skipping this batch.")
            return None 

        # 2. Lightweight guard: Filter items if encoder_input is empty (should be rare due to __getitem__ checks)
        #    Also check for empty decoder_target if it implies an invalid sample for the model.
        #    For now, models are often robust to empty targets if the loss function handles it (e.g. ignore_index for padding)
        #    or if it means "predict EOS immediately". Let's primarily ensure encoder_input is valid.
        
        processed_batch = []
        for i, item in enumerate(batch):
            if not isinstance(item, dict) or 'encoder_input' not in item or 'decoder_target' not in item:
                logger.warning(f"SFTDataset.collate_fn: Item {i} is not a valid dict or missing keys. Skipping. Item: {str(item)[:100]}")
                continue
            if item['encoder_input'].numel() == 0:
                logger.debug(f"SFTDataset.collate_fn: Item {i} has empty 'encoder_input' tensor. Skipping item.")
                continue
            # Optionally, add a check for empty decoder_target if it's strictly invalid for the model
            # e.g. if item['decoder_target'].numel() == 0 and not model_can_handle_empty_target:
            #    logger.debug(f"SFTDataset.collate_fn: Item {i} has empty 'decoder_target'. Skipping.")
            #    continue
            processed_batch.append(item)
        
        batch = processed_batch
        if not batch:
            if original_batch_len > 0: # if some items existed before this collator-specific filtering
                 logger.warning("SFTDataset.collate_fn: Batch is empty after collator's internal filtering (e.g. empty tensors). Skipping batch.")
            return None

        # Proceed with padding and batching valid items
        max_enc_len = max(item['encoder_input'].size(0) for item in batch)
        max_dec_len = max(item['decoder_target'].size(0) for item in batch if item['decoder_target'].numel() > 0) if any(item['decoder_target'].numel() > 0 for item in batch) else 0

        current_batch_size = len(batch)
        
        # Initialize tensors with the provided pad_id
        encoder_input_batch = torch.full((current_batch_size, max_enc_len), pad_id, dtype=torch.long)
        encoder_attention_mask_batch = torch.zeros(current_batch_size, max_enc_len, dtype=torch.bool)
        decoder_target_batch = torch.full((current_batch_size, max_dec_len), pad_id, dtype=torch.long)
        relation_matrix_batch = torch.zeros(current_batch_size, max_enc_len, max_enc_len, dtype=torch.long)
        schema_mask_batch = torch.zeros(current_batch_size, max_enc_len, dtype=torch.bool)
        
        for i, item in enumerate(batch):
            enc_len = item['encoder_input'].size(0)
            encoder_input_batch[i, :enc_len] = item['encoder_input']
            encoder_attention_mask_batch[i, :enc_len] = item['encoder_attention_mask']
            
            # Relation matrix and schema mask checks (shape based on enc_len)
            if 'relation_matrix' in item and item['relation_matrix'].ndim == 2 and\
               item['relation_matrix'].shape[0] == enc_len and item['relation_matrix'].shape[1] == enc_len:
                relation_matrix_batch[i, :enc_len, :enc_len] = item['relation_matrix']
            # else: (Optional: log if shape mismatch and non-default tensor was provided)

            if 'schema_mask' in item and item['schema_mask'].ndim == 1 and item['schema_mask'].shape[0] == enc_len:
                 schema_mask_batch[i, :enc_len] = item['schema_mask']
            # else: (Optional: log if shape mismatch and non-default tensor was provided)

            dec_len = item['decoder_target'].size(0)
            if dec_len > 0:
                decoder_target_batch[i, :dec_len] = item['decoder_target']
        
        return {
            'encoder_input': encoder_input_batch,
            'decoder_target': decoder_target_batch,
            'encoder_attention_mask': encoder_attention_mask_batch,
            'relation_matrix': relation_matrix_batch,
            'schema_mask': schema_mask_batch
        }
