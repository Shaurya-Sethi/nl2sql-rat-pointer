import torch
from torch.utils.data import Dataset
import logging
from typing import List, Optional
from tokenizer import NL2SQLTokenizer

logger = logging.getLogger(__name__)

class PretrainingDataset(Dataset):
    def __init__(self, data_file: str, tokenizer: NL2SQLTokenizer, max_len: int = 512, pad_id: int = 18):
        """
        Initialize the pretraining dataset.
        
        Args:
            data_file (str): Path to the data file (space-separated token IDs per line)
            tokenizer: Tokenizer instance (for vocab_size, special tokens)
            max_len (int): Maximum sequence length for pretraining.
            pad_id (int): Token ID for padding.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.min_len = 2 # Pretrain rule: len < 2 is too short
        self.pad_token_id = pad_id

        try:
            self.vocab_size = self.tokenizer.vocab_size
        except AttributeError:
            try:
                self.vocab_size = self.tokenizer.get_vocab_size()
            except AttributeError:
                logger.error("PretrainingDataset: Tokenizer has no 'vocab_size' or 'get_vocab_size()'. OOV checks will be skipped.")
                self.vocab_size = float('inf') # Effectively skips OOV check
        
        # Get special token IDs for pretrain truncation rule (preserve <SQL> and </SQL>)
        try:
            self.sql_token_id = self.tokenizer.get_special_token_id('SQL_START') # Use defined name
            self.sql_end_token_id = self.tokenizer.get_special_token_id('SQL_END') # Use defined name
        except KeyError as e:  # Catch specifically KeyError if a name is not found
            logger.warning(f"PretrainingDataset: Could not get SQL_START or SQL_END token IDs from tokenizer: {e}. Pretrain truncation might not preserve them.")
            self.sql_token_id = None
            self.sql_end_token_id = None

        self.dropped_count = 0
        self.truncated_count = 0
        self._summary_logged_first_time = False
        self.last_logged_dropped_count = 0
        self.last_logged_truncated_count = 0

        logger.info(f"PretrainingDataset: Initializing with data_file='{data_file}', max_len={self.max_len}, min_len={self.min_len}, pad_id={self.pad_token_id}")
        self.raw_examples: List[List[int]] = [] # Store token_ids directly
        raw_lines_count = 0
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line_str in enumerate(f):
                raw_lines_count +=1
                line_str = line_str.strip()
                if not line_str:
                    logger.debug(f"PretrainingDataset: Skipping empty line {line_num+1} in {data_file} at load time.")
                    self.dropped_count += 1
                    continue
                try:
                    # Parse directly to List[int]
                    token_ids = [int(token_id_str) for token_id_str in line_str.split()]
                    if not token_ids: # Line had only whitespace
                        logger.debug(f"PretrainingDataset: Skipping line {line_num+1} (whitespace only) in {data_file} at load time.")
                        self.dropped_count +=1
                        continue
                    self.raw_examples.append(token_ids)
                except ValueError as e:
                    logger.warning(f"PretrainingDataset: Error parsing token IDs (non-integer) on line {line_num+1} in {data_file}: {e}. Line: '{line_str[:100]}...' Skipping.")
                    self.dropped_count += 1
                    continue
        
        logger.info(f"PretrainingDataset: Loaded {len(self.raw_examples)} examples from {raw_lines_count} raw lines. Initial drops (empty/parse error): {self.dropped_count}")
        self._log_counts_summary_if_needed(dataset_name="PretrainingDataset", force_log=True)

    def _log_counts_summary_if_needed(self, dataset_name: str, force_log: bool = False):
        current_dropped = self.dropped_count
        current_truncated = self.truncated_count
        
        has_significant_counts = (current_dropped > 0 or current_truncated > 0)
        counts_changed = (current_dropped != self.last_logged_dropped_count or 
                          current_truncated != self.last_logged_truncated_count)

        if has_significant_counts and (force_log or not self._summary_logged_first_time or counts_changed):
            # Total initial lines = len(self.raw_examples) + self.dropped_count (for those dropped at load time)
            total_processed_or_loaded = len(self.raw_examples) + sum(1 for ex in self.raw_examples if ex is None) # Approximation
            logger.info(f"{dataset_name} Counts: Dropped={current_dropped}, Truncated={current_truncated}. (Initially loaded: {len(self.raw_examples) + self.dropped_count})")
            self.last_logged_dropped_count = current_dropped
            self.last_logged_truncated_count = current_truncated
            self._summary_logged_first_time = True

    def __len__(self):
        self._log_counts_summary_if_needed(dataset_name="PretrainingDataset", force_log=True)
        return len(self.raw_examples)

    def __getitem__(self, idx):
        if idx >= len(self.raw_examples):
            raise IndexError(f"Index {idx} out of range for PretrainingDataset with {len(self.raw_examples)} examples.")
        
        original_token_ids = self.raw_examples[idx]
        # Make a copy for modification, as self.raw_examples[idx] should remain pristine
        token_ids = list(original_token_ids) 

        # 1. Trim leading/trailing pad_ids
        start_trim_idx = 0
        while start_trim_idx < len(token_ids) and token_ids[start_trim_idx] == self.pad_token_id:
            start_trim_idx += 1
        end_trim_idx = len(token_ids)
        while end_trim_idx > start_trim_idx and token_ids[end_trim_idx - 1] == self.pad_token_id:
            end_trim_idx -= 1
        token_ids = token_ids[start_trim_idx:end_trim_idx]

        # 2. Rule: Empty / All tokens equal to pad_id
        if not token_ids:
            logger.debug(f"PretrainingDataset [EX {idx}]: Sequence empty after trimming pad_id ({self.pad_token_id}). Dropping.")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="PretrainingDataset")
            return None 

        # 3. Rule: OOV ID (token < 0 or token >= vocab_size)
        if self.vocab_size != float('inf'): # Only if vocab_size is known
            for i, token_id_val in enumerate(token_ids):
                if not (0 <= token_id_val < self.vocab_size):
                    logger.debug(f"PretrainingDataset [EX {idx}]: OOV token ID {token_id_val} at pos {i} (vocab size {self.vocab_size}). Dropping. Sample: {token_ids[:10]}...")
                    self.dropped_count += 1
                    self._log_counts_summary_if_needed(dataset_name="PretrainingDataset")
                    return None 
        
        # 4. Rule: Too short (Pretrain: len < 2)
        if len(token_ids) < self.min_len:
            logger.debug(f"PretrainingDataset [EX {idx}]: Sequence too short (len {len(token_ids)} < min_len {self.min_len}) after trimming. Dropping. Sample: {token_ids}")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="PretrainingDataset")
            return None 

        # 5. Rule: Too long (Pretrain > 512 -> truncate, keep <SQL> and </SQL> if present)
        if len(token_ids) > self.max_len:
            self.truncated_count += 1
            # self._log_counts_summary_if_needed(dataset_name="PretrainingDataset") # Logged by other means
            
            first_token = token_ids[0] if token_ids else None
            last_token = token_ids[-1] if token_ids else None
            
            token_ids = token_ids[:self.max_len]
            
            # Ensure <SQL> is first token if it was originally and max_len > 0
            if self.sql_token_id is not None and first_token == self.sql_token_id and self.max_len > 0:
                if token_ids[0] != self.sql_token_id:
                     # This case should be rare if max_len is reasonably small and <SQL> is present
                     # If max_len is 1, and it was <SQL> ... </SQL>, it becomes just <SQL>
                     token_ids[0] = self.sql_token_id 
            
            # Ensure </SQL> is last token if it was originally and max_len > (1 if <SQL> also present, else 0)
            if self.sql_end_token_id is not None and last_token == self.sql_end_token_id and self.max_len > 0:
                # Check if space is available for </SQL> token
                # If <SQL> is present and also preserved, it takes index 0.
                # So </SQL> must be at an index > 0 if both are present.
                min_idx_for_end_sql = 0
                if self.sql_token_id is not None and first_token == self.sql_token_id and token_ids[0] == self.sql_token_id:
                    min_idx_for_end_sql = 1
                
                if self.max_len > min_idx_for_end_sql: # Ensure there's space for it
                    if token_ids[self.max_len - 1] != self.sql_end_token_id:
                         token_ids[self.max_len - 1] = self.sql_end_token_id
                    # If <SQL> and </SQL> were same, and max_len is 1, token_ids[0] is already <SQL> (or </SQL> if it's the same ID)
                    # If <SQL> took pos 0, and max_len is e.g. 2, pos 1 gets </SQL>
                    # This logic ensures that if <SQL> ... </SQL> was truncated, and both tokens are distinct and required,
                    # they are preserved if max_len >= 2. If max_len = 1, only first token is preserved.
                    # If SQL_START and SQL_END are the same token ID, this logic is simpler.
            logger.debug(f"PretrainingDataset [EX {idx}]: Sequence truncated to {len(token_ids)} (max_len {self.max_len}).")
        
        # If after truncation, the sequence somehow became too short (e.g. max_len=1, and original was <SQL>, </SQL>)
        # Re-check min_len. This is important if max_len is very small.
        if len(token_ids) < self.min_len:
            logger.debug(f"PretrainingDataset [EX {idx}]: Sequence became too short (len {len(token_ids)}) after truncation. Dropping. Original: {original_token_ids[:10]}")
            self.dropped_count += 1
            # self._log_counts_summary_if_needed(dataset_name="PretrainingDataset")
            return None

        input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        attention_mask_tensor = torch.ones_like(input_ids_tensor, dtype=torch.bool)
        seq_len = input_ids_tensor.size(0)
        
        # Dummies for keys expected by the Trainer (if any)
        # As per original code, relation_matrix and schema_mask are dummies.
        relation_matrix_dummy = torch.zeros((seq_len, seq_len), dtype=torch.long)
        schema_mask_dummy = torch.zeros(seq_len, dtype=torch.bool)
        
        return {
            'encoder_input': input_ids_tensor,             # For LM, input is often same as target
            'encoder_attention_mask': attention_mask_tensor,
            'decoder_target': input_ids_tensor.clone(),    # Labels = inputs for LM pretraining
            'relation_matrix': relation_matrix_dummy,     # Dummy for pretrain
            'schema_mask': schema_mask_dummy,             # Dummy for pretrain
        }

    @staticmethod
    def collate_fn(batch, pad_id=18): # Default pad_id from user context
        original_batch_len = len(batch)
        # 1. Filter out None items (samples dropped by __getitem__)
        batch = [item for item in batch if item is not None]
        
        if not batch:
            if original_batch_len > 0:
                logger.warning("PretrainingDataset.collate_fn: Batch is empty after __getitem__ filtering. Skipping batch.")
            return None 

        # 2. Lightweight guard: filter items if encoder_input is somehow empty (should be caught by __getitem__ min_len)
        processed_batch = []
        for i, item in enumerate(batch):
            if not isinstance(item, dict) or 'encoder_input' not in item or item['encoder_input'].numel() == 0:
                logger.debug(f"PretrainingDataset.collate_fn: Item {i} is invalid (not dict, or empty 'encoder_input'). Skipping. Item: {str(item)[:100]}")
                continue
            processed_batch.append(item)
        
        batch = processed_batch
        if not batch:
            if original_batch_len > 0:
                 logger.warning("PretrainingDataset.collate_fn: Batch is empty after collator's internal filtering. Skipping batch.")
            return None

        # Get max length in the valid batch for padding
        current_max_len = max(item['encoder_input'].size(0) for item in batch)
        current_batch_size = len(batch)

        # Initialize tensors with the provided pad_id
        encoder_input_batch = torch.full((current_batch_size, current_max_len), pad_id, dtype=torch.long)
        encoder_attention_mask_batch = torch.zeros(current_batch_size, current_max_len, dtype=torch.bool)
        decoder_target_batch = torch.full((current_batch_size, current_max_len), pad_id, dtype=torch.long)
        # Dummies also need to be batched, using current_max_len
        relation_matrix_batch = torch.zeros((current_batch_size, current_max_len, current_max_len), dtype=torch.long)
        schema_mask_batch = torch.zeros((current_batch_size, current_max_len), dtype=torch.bool)

        for i, item in enumerate(batch):
            length = item['encoder_input'].size(0)
            encoder_input_batch[i, :length] = item['encoder_input']
            encoder_attention_mask_batch[i, :length] = item['encoder_attention_mask']
            decoder_target_batch[i, :length] = item['decoder_target']
            # Pad dummy relation_matrix and schema_mask correctly up to item's actual length
            # The dummy tensors from __getitem__ are (length, length) and (length,)
            if item['relation_matrix'].ndim == 2 and item['relation_matrix'].shape[0] == length:
                 relation_matrix_batch[i, :length, :length] = item['relation_matrix'] # This is item-specific dummy
            if item['schema_mask'].ndim == 1 and item['schema_mask'].shape[0] == length:
                 schema_mask_batch[i, :length] = item['schema_mask'] # This is item-specific dummy

        return {
            'encoder_input': encoder_input_batch,
            'encoder_attention_mask': encoder_attention_mask_batch,
            'decoder_target': decoder_target_batch,
            'relation_matrix': relation_matrix_batch, # Batched dummy
            'schema_mask': schema_mask_batch,       # Batched dummy
        }
