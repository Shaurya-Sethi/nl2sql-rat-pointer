DEBUG_VERBOSE = False  # Set True to enable deep per-sample debug logging
import os
import logging
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from tokenizer import NL2SQLTokenizer
from config import NL2SQLConfig
from relation_matrix import RelationMatrixBuilder

logger = logging.getLogger(__name__)

def debug_log(msg, logger=None):
    """Helper function for debug logging that respects DEBUG_VERBOSE flag"""
    if DEBUG_VERBOSE:
        if logger:
            logger.debug(msg)
        else:
            print(msg)

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning of NL2SQL model.
    Assumes data_file is a .txt file where each line contains space-separated
    integer token IDs representing a full concatenated sequence:
    <NL> ... </NL> <SCHEMA> ... </SCHEMA> <EXT> ... </EXT> <COT> ... </COT> <SQL> ... </SQL>
    
    Implements lazy loading: reads line offsets at initialization and processes lines on demand.
    """
    
    _source_truncation_warning_logged = False # Static class variable for one-time warning
    _first_sample_globally_logged = False  # Class variable for one-time verification logging
    
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
        self.data_file_path = data_file # Store path for __getitem__
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
        
        # Get special token IDs for sequence splitting and truncation preservation
        self.cot_start_token_id = self.tokenizer.get_special_token_id('COT_START')
        self.sql_end_token_id = self.tokenizer.get_special_token_id('SQL_END')
        
        # For SFT truncation, we want to preserve SQL_END if it's the last token
        self.sft_end_special_tokens = {self.sql_end_token_id}
        logger.info(f"SFTDataset: Using {{ {self.sql_end_token_id}: '{self.tokenizer.special_tokens.get('SQL_END')}' }} as special end tokens for SFT truncation preservation.")

        self.dropped_count = 0
        self.truncated_count = 0
        self._summary_logged_first_time = False 
        self.last_logged_dropped_count = 0
        self.last_logged_truncated_count = 0

        logger.info(f"SFTDataset: Initializing with data_file='{data_file}', max_len={self.max_len}, min_len={self.min_len}, pad_id={self.pad_token_id}")
        
        self.line_offsets: List[int] = []
        self.num_lines = 0
        initial_raw_lines = 0
        initial_empty_or_whitespace_drops = 0

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"SFTDataset: Data file not found: {data_file}")

        logger.info(f"SFTDataset: Scanning '{data_file}' to build line offsets...")
        with open(data_file, 'rb') as f: # Open in binary mode to get byte offsets
            while True:
                offset = f.tell()
                line_bytes = f.readline()
                initial_raw_lines += 1
                if not line_bytes:
                    initial_raw_lines -=1 # Last readline is empty on EOF
                    break
                # Minimal check for truly empty or whitespace-only lines during offset scan
                # More robust checks will happen in __getitem__
                try:
                    line_str = line_bytes.decode('utf-8').strip()
                    if not line_str:
                        initial_empty_or_whitespace_drops += 1
                        # We don't add this line to offsets if it's completely empty/whitespace
                        continue 
                except UnicodeDecodeError:
                    logger.warning(f"SFTDataset: UnicodeDecodeError on line {self.num_lines + 1} during initial scan. This line might be skipped or cause issues later.")
                    # We'll still record its offset and let __getitem__ handle the error
                    pass # Continue to add offset, let __getitem__ handle parsing error
                
                self.line_offsets.append(offset)
                self.num_lines += 1
        
        self.dropped_count += initial_empty_or_whitespace_drops # Count lines dropped during initial scan
        logger.info(f"SFTDataset: Scan complete. Found {self.num_lines} processable lines (offsets recorded). Total raw lines scanned: {initial_raw_lines}. Initial empty/whitespace drops from scan: {initial_empty_or_whitespace_drops}.")
        self._log_counts_summary_if_needed(dataset_name="SFTDataset", force_log=True)

    def _log_counts_summary_if_needed(self, dataset_name: str, force_log: bool = False):
        current_dropped = self.dropped_count
        current_truncated = self.truncated_count
        
        has_significant_counts = (current_dropped > 0 or current_truncated > 0)
        counts_changed = (current_dropped != self.last_logged_dropped_count or 
                          current_truncated != self.last_logged_truncated_count)

        if has_significant_counts and (force_log or not self._summary_logged_first_time or counts_changed):
            logger.info(f"{dataset_name} Counts: Dropped={current_dropped}, Truncated={current_truncated}. (Lines with offsets: {self.num_lines})")
            self.last_logged_dropped_count = current_dropped
            self.last_logged_truncated_count = current_truncated
            self._summary_logged_first_time = True

    def __len__(self):
        self._log_counts_summary_if_needed(dataset_name="SFTDataset", force_log=not self._summary_logged_first_time)
        return self.num_lines # Number of lines with recorded offsets

    def __getitem__(self, idx):
        if not (0 <= idx < self.num_lines):
             raise IndexError(f"Index {idx} out of range for SFTDataset with {self.num_lines} lines.")

        try:
            # Open file, seek, read line, then close. This is safer for multiprocessing.
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                f.seek(self.line_offsets[idx])
                line_str = f.readline()
        except Exception as e:
            logger.error(f"SFTDataset [EX {idx}]: Error reading line from file at offset {self.line_offsets[idx]}: {e}")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None

        line_str = line_str.strip()
        if not line_str:
            debug_log(f"SFTDataset [EX {idx}]: Line is empty or whitespace after read. Dropping.", logger)
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None
            
        try:
            original_token_ids = [int(token_id_str) for token_id_str in line_str.split()]
            if not original_token_ids:
                debug_log(f"SFTDataset [EX {idx}]: Line resulted in empty token list after split. Dropping.", logger)
                self.dropped_count += 1
                self._log_counts_summary_if_needed(dataset_name="SFTDataset")
                return None
        except ValueError as e:
            logger.warning(f"SFTDataset [EX {idx}]: Error parsing token IDs (non-integer): {e}. Line: '{line_str[:100]}...' Dropping.")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None
        
        token_ids = list(original_token_ids) # Work with a copy
        
        # 1. Trim leading/trailing pad_ids
        start_idx = 0
        while start_idx < len(token_ids) and token_ids[start_idx] == self.pad_token_id:
            start_idx += 1
        end_idx = len(token_ids)
        while end_idx > start_idx and token_ids[end_idx - 1] == self.pad_token_id:
            end_idx -= 1
        token_ids = token_ids[start_idx:end_idx]

        # 2. Rule: Empty / All tokens equal to pad_id
        if not token_ids:
            debug_log(f"SFTDataset [EX {idx}]: Sequence empty after trimming pad_id ({self.pad_token_id}). Dropping.", logger)
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None

        # 3. Rule: OOV ID (token < 0 or token >= vocab_size)
        if self.vocab_size != float('inf'): # Only if vocab_size is known
            for i, token_id in enumerate(token_ids):
                if not (0 <= token_id < self.vocab_size):
                    debug_log(f"SFTDataset [EX {idx}]: OOV token ID {token_id} at pos {i} (vocab size {self.vocab_size}). Dropping. Sample: {token_ids[:10]}...", logger)
                    self.dropped_count += 1
                    self._log_counts_summary_if_needed(dataset_name="SFTDataset")
                    return None
        
        # 4. Rule: Too short (SFT: len < 4)
        if len(token_ids) < self.min_len:
            debug_log(f"SFTDataset [EX {idx}]: Sequence too short (len {len(token_ids)} < min_len {self.min_len}) after trimming. Dropping. Sample: {token_ids}", logger)
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None

        # 5. Find COT_START and SQL_END tokens for splitting encoder input and decoder target
        try:
            cot_start_idx = token_ids.index(self.cot_start_token_id)
        except ValueError:
            logger.warning(f"SFTDataset [EX {idx}]: COT_START token not found. Dropping sample.")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None

        try:
            sql_end_idx = token_ids.index(self.sql_end_token_id)
            if sql_end_idx <= cot_start_idx:
                logger.warning(f"SFTDataset [EX {idx}]: SQL_END token appears before or at COT_START. Invalid sequence. Dropping sample.")
                self.dropped_count += 1
                self._log_counts_summary_if_needed(dataset_name="SFTDataset")
                return None
        except ValueError:
            logger.warning(f"SFTDataset [EX {idx}]: SQL_END token not found. Dropping sample.")
            self.dropped_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")
            return None

        # Split into encoder input and decoder target
        raw_encoder_input_tokens = token_ids[:cot_start_idx]  # Up to but not including COT_START
        raw_target_tokens = token_ids[cot_start_idx:sql_end_idx + 1]  # From COT_START through SQL_END

        # 6. Rule: Too long (SFT > max_len (e.g. 2048 for full sequence))
        original_len_before_max_len_trunc = len(token_ids)
        perform_max_len_truncation = False
        if len(token_ids) > self.max_len:
            perform_max_len_truncation = True
            original_last_token_if_truncated = token_ids[-1] # Before truncating token_ids
            token_ids = token_ids[:self.max_len]
            # Preserve special end token if it was the original last token and got cut off
            if self.max_len > 0 and original_last_token_if_truncated in self.sft_end_special_tokens and \
               token_ids[-1] != original_last_token_if_truncated:
                token_ids[-1] = original_last_token_if_truncated 
            logger.debug(f"SFTDataset [EX {idx}]: Full sequence truncated from {original_len_before_max_len_trunc} to {len(token_ids)} (max_len {self.max_len}).")
            
            # After truncation, check if we still have valid COT_START and SQL_END in sequence
            try:
                new_cot_start_idx = token_ids.index(self.cot_start_token_id)
                new_sql_end_idx = token_ids.index(self.sql_end_token_id)
                if new_sql_end_idx <= new_cot_start_idx:
                    logger.warning(f"SFTDataset [EX {idx}]: After truncation, SQL_END appears before or at COT_START. Dropping sample.")
                    self.dropped_count += 1
                    self.truncated_count += 1
                    self._log_counts_summary_if_needed(dataset_name="SFTDataset")
                    return None
                # Update splits after truncation
                raw_encoder_input_tokens = token_ids[:new_cot_start_idx]
                raw_target_tokens = token_ids[new_cot_start_idx:new_sql_end_idx + 1]
            except ValueError:
                logger.warning(f"SFTDataset [EX {idx}]: After truncation, COT_START or SQL_END not found. Dropping sample.")
                self.dropped_count += 1
                self.truncated_count += 1
                self._log_counts_summary_if_needed(dataset_name="SFTDataset")
                return None

        if perform_max_len_truncation:
            self.truncated_count += 1
            self._log_counts_summary_if_needed(dataset_name="SFTDataset")

        # Pointer-Generator (PG) specific truncation for the source part (everything before COT)
        final_encoder_input_tokens = raw_encoder_input_tokens
        sft_pg_source_max_len = self.config.phase_max_len_pg
        if self.config.use_pointer_generator and sft_pg_source_max_len is not None and \
           len(raw_encoder_input_tokens) > sft_pg_source_max_len:
            if not SFTDataset._source_truncation_warning_logged:
                logger.warning(
                    f"SFTDataset: PG source (pre-COT part) length {len(raw_encoder_input_tokens)} exceeds phase_max_len_pg {sft_pg_source_max_len}. "
                    f"Truncating source part from the beginning. This warning is shown once per class instance."
                )
                SFTDataset._source_truncation_warning_logged = True
            
            amount_to_cut = len(raw_encoder_input_tokens) - sft_pg_source_max_len
            final_encoder_input_tokens = raw_encoder_input_tokens[amount_to_cut:]
            logger.debug(f"SFTDataset [EX {idx}]: PG source part truncated from {len(raw_encoder_input_tokens)} to {len(final_encoder_input_tokens)} (PG source max: {sft_pg_source_max_len}).")

        if not final_encoder_input_tokens:
             logger.debug(f"SFTDataset [EX {idx}]: final_encoder_input_tokens became empty after PG truncation. Dropping. Original line: {original_token_ids[:30]}")
             self.dropped_count +=1
             self._log_counts_summary_if_needed(dataset_name="SFTDataset")
             return None
        
        # Create tensors
        input_ids = torch.tensor(final_encoder_input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.tensor(raw_target_tokens, dtype=torch.long)

        schema_meta: List = []
        relation_matrix_dim = len(final_encoder_input_tokens)
        relation_matrix = torch.zeros((relation_matrix_dim, relation_matrix_dim), dtype=torch.long)
        schema_mask_for_pg = torch.zeros_like(input_ids, dtype=torch.bool)

        if self.config.use_pointer_generator and relation_matrix_dim > 0:
            # --- NEW: Extract just the schema segment from encoder input ---
            schema_start_id = self.tokenizer.get_special_token_id("SCHEMA_START")
            schema_end_id = self.tokenizer.get_special_token_id("SCHEMA_END")
            try:
                schema_start = final_encoder_input_tokens.index(schema_start_id)
                schema_end = final_encoder_input_tokens.index(schema_end_id)
                # Extract schema tokens (including <SCHEMA> and </SCHEMA> tokens; adjust as needed for your parser)
                schema_tokens = final_encoder_input_tokens[schema_start:schema_end + 1]
                # --- DEBUG LOGGING ---
                if DEBUG_VERBOSE:
                    logger.debug(f"[PG DEBUG][idx={idx}] Extracted schema_tokens (IDs): {schema_tokens}")
                    try:
                        detok_schema = self.tokenizer.decode(schema_tokens, skip_special_tokens=False)
                    except Exception as e:
                        detok_schema = f"<decode error: {e}>"
                    logger.debug(f"[PG DEBUG][idx={idx}] Detokenized schema segment: '{detok_schema}'")
                    logger.debug(f"[PG DEBUG][idx={idx}] Full encoder input tokens: {final_encoder_input_tokens}")
                    try:
                        detok_full = self.tokenizer.decode(final_encoder_input_tokens, skip_special_tokens=False)
                    except Exception as e:
                        detok_full = f"<decode error: {e}>"
                    logger.debug(f"[PG DEBUG][idx={idx}] Detokenized full encoder input: '{detok_full}'")
            except ValueError:
                logger.warning(f"SFTDataset [EX {idx}]: SCHEMA_START or SCHEMA_END not found in encoder input. Skipping PG schema parse.")
                schema_tokens = []

            try:
                # Parse directly from the full encoder sequence so spans match
                schema_meta = self.relation_builder.parse_schema_tokens(
                    final_encoder_input_tokens
                )
                # Let the builder use the pre-parsed tokens (or omit the arg to
                # have it re-parse internally)
                relation_matrix = self.relation_builder.build_relation_matrix(
                    final_encoder_input_tokens,
                    schema_meta,          # ← optional but keeps one parse
                )
                
                for token_s in schema_meta:
                    start_idx_s, end_idx_s = token_s.span_start, token_s.span_end
                    if start_idx_s < relation_matrix_dim and end_idx_s < relation_matrix_dim:
                        schema_mask_for_pg[start_idx_s : end_idx_s + 1] = True
                    elif start_idx_s < relation_matrix_dim:
                        schema_mask_for_pg[start_idx_s : relation_matrix_dim] = True
            except Exception as e:
                logger.error(f"SFTDataset [EX {idx}]: Error building relation matrix/schema mask: {e}", exc_info=True)
                relation_matrix = torch.zeros((relation_matrix_dim, relation_matrix_dim), dtype=torch.long)
                schema_mask_for_pg = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Verify first sample globally (for debugging)
        if DEBUG_VERBOSE and not SFTDataset._first_sample_globally_logged:
            try:
                debug_log("\nFirst sample verification (SFTDataset - lazy loaded):", logger)
                debug_log(f"Original line (idx {idx}): '{line_str[:100]}...'", logger)
                debug_log(f"Encoder input tokens (before COT) (len={len(final_encoder_input_tokens)}): {self.tokenizer.decode(final_encoder_input_tokens)}", logger)
                debug_log(f"Decoder target tokens (COT through SQL_END) (len={len(raw_target_tokens)}): {self.tokenizer.decode(raw_target_tokens)}", logger)
                if self.config.use_pointer_generator:
                    debug_log(f"Schema mask for PG (sum of True): {schema_mask_for_pg.sum().item()}", logger)
                SFTDataset._first_sample_globally_logged = True
            except Exception as e:
                logger.error(f"Error logging first sample: {e}")

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
        Collate function for SFTDataset. Pads and batches items, handling edge cases robustly.
        - Filters out None items and invalid dicts.
        - Skips items with empty encoder_input.
        - Always ensures decoder_target has at least length 1 (prevents shape errors).
        - Pads all fields to batch max lengths.
        """
        original_batch_len = len(batch)
        # 1. Filter out None items (samples dropped by __getitem__)
        batch = [item for item in batch if item is not None]
        if not batch:
            if original_batch_len > 0:
                logger.warning("SFTDataset.collate_fn: Batch is empty after __getitem__ filtering. Skipping this batch.")
            return None 

        # 2. Filter out items with missing/invalid keys or empty encoder_input
        processed_batch = []
        for i, item in enumerate(batch):
            if not isinstance(item, dict) or 'encoder_input' not in item or 'decoder_target' not in item:
                logger.warning(f"SFTDataset.collate_fn: Item {i} is not a valid dict or missing keys. Skipping. Item: {str(item)[:100]}")
                continue
            if item['encoder_input'].numel() == 0:
                logger.debug(f"SFTDataset.collate_fn: Item {i} has empty 'encoder_input' tensor. Skipping item.")
                continue
            processed_batch.append(item)

        batch = processed_batch
        if not batch:
            if original_batch_len > 0:
                logger.warning("SFTDataset.collate_fn: Batch is empty after collator's internal filtering (e.g. empty tensors). Skipping batch.")
            return None

        # --- Padding logic ---
        max_enc_len = max(item['encoder_input'].size(0) for item in batch)
        # Always ensure at least length 1 for decoder target (prevents shape errors)
        max_dec_len = max([item['decoder_target'].size(0) if item['decoder_target'].numel() > 0 else 1 for item in batch])

        current_batch_size = len(batch)

        encoder_input_batch = torch.full((current_batch_size, max_enc_len), pad_id, dtype=torch.long)
        encoder_attention_mask_batch = torch.zeros(current_batch_size, max_enc_len, dtype=torch.bool)
        decoder_target_batch = torch.full((current_batch_size, max_dec_len), pad_id, dtype=torch.long)
        relation_matrix_batch = torch.zeros(current_batch_size, max_enc_len, max_enc_len, dtype=torch.long)
        schema_mask_batch = torch.zeros(current_batch_size, max_enc_len, dtype=torch.bool)

        for i, item in enumerate(batch):
            enc_len = item['encoder_input'].size(0)
            encoder_input_batch[i, :enc_len] = item['encoder_input']
            encoder_attention_mask_batch[i, :enc_len] = item['encoder_attention_mask'] 

            if 'relation_matrix' in item and item['relation_matrix'].ndim == 2 and \
               item['relation_matrix'].shape[0] == enc_len and item['relation_matrix'].shape[1] == enc_len:
                relation_matrix_batch[i, :enc_len, :enc_len] = item['relation_matrix']

            if 'schema_mask' in item and item['schema_mask'].ndim == 1 and item['schema_mask'].shape[0] == enc_len:
                schema_mask_batch[i, :enc_len] = item['schema_mask']

            dec_len = item['decoder_target'].size(0)
            # If decoder_target is empty, leave row as all PADs
            if dec_len > 0:
                decoder_target_batch[i, :dec_len] = item['decoder_target']

        return {
            'encoder_input': encoder_input_batch,
            'decoder_target': decoder_target_batch,
            'encoder_attention_mask': encoder_attention_mask_batch,
            'relation_matrix': relation_matrix_batch,
            'schema_mask': schema_mask_batch
        }
