import torch
import logging
import re
from typing import List, Dict, Tuple, Optional, Set, Iterator
from dataclasses import dataclass
from tokenizer import NL2SQLTokenizer

logger = logging.getLogger(__name__)

@dataclass
class SchemaToken:
    """Metadata for a schema token span in the *full* input sequence."""
    span_start: int        # inclusive index in full token list
    span_end:   int        # inclusive index in full token list
    token_type: str        # 'table' | 'column' | 'pk' | 'fk'
    table_name:   Optional[str] = None
    column_name:  Optional[str] = None
    references:   Optional[Tuple[str, str]] = None   # (ref_table, ref_column)
    
    def __str__(self) -> str:
        """String representation for easier debugging."""
        refs = f" -> {self.references[0]}.{self.references[1]}" if self.references else ""
        return f"{self.token_type}[{self.span_start}:{self.span_end}] {self.table_name}.{self.column_name}{refs}"


class RelationMatrixBuilder:
    """
    Builds a seq_len × seq_len matrix of relation-type IDs.
    Relation IDs:
        0: no_relation / padding (default)
        1: same_table
        2: pk_fk  (PK ↔ FK)
        3: table_column
        4: same_column (name match across tables)
    """

    _REL_MAP = {
        'no_relation': 0,
        'same_table' : 1,
        'pk_fk'      : 2,
        'table_column': 3,
        'same_column': 4,
    }

    def __init__(self, tokenizer: NL2SQLTokenizer, num_relations: int = 5):
        """
        Initialize relation matrix builder.
        
        Args:
            tokenizer: NL2SQLTokenizer instance
            num_relations: Number of relation types
        """
        try:
            self.sp = tokenizer.sp
            self.tok_id = tokenizer.special_token_ids
            
            # Cache for decoded tokens to improve performance
            self._decode_cache = {}
            self._cache_hits = 0
            self._cache_misses = 0

            # Validate special tokens are present in the provided tokenizer's mapping
            required_token_names = {
                'SCHEMA_START', 'SCHEMA_END',
                'PK_START', 'PK_END',
                'FK_START', 'FK_END'
            }
            missing_tokens = required_token_names - set(self.tok_id.keys())
            if missing_tokens:
                raise ValueError(f"Missing required special token names in tokenizer's special_token_ids: {missing_tokens}")
                
            self.num_relations = num_relations
            
            if num_relations < len(self._REL_MAP):
                raise ValueError(f"num_relations={num_relations} must be at least {len(self._REL_MAP)}")
                
        except Exception as e:
            logger.error(f"Error initializing RelationMatrixBuilder: {e}")
            raise

    def _decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to string with caching for performance.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        """
        if not ids:
            return ""
            
        # Use tuple as cache key since lists are not hashable
        cache_key = tuple(ids)
        
        if cache_key in self._decode_cache:
            self._cache_hits += 1
            return self._decode_cache[cache_key]
        
        try:
            result = self.sp.decode(ids)
            self._decode_cache[cache_key] = result
            self._cache_misses += 1
            
            # Prevent cache from growing too large
            if len(self._decode_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._decode_cache.keys())[:100]
                for key in oldest_keys:
                    del self._decode_cache[key]
                    
            return result
        except Exception as e:
            logger.error(f"Error decoding tokens {ids}: {e}")
            return ""

    def _normalize_name(self, name: str) -> str:
        """
        Normalize database identifier name.
        
        Args:
            name: Raw name string
            
        Returns:
            Normalized name
        """
        if not name:
            return ""
            
        # Remove whitespace and lowercase
        normalized = re.sub(r'\s+', ' ', name.strip()).lower()
        
        # Remove various quote types and brackets
        normalized = re.sub(r'^["`\'\[\]]+|["`\'\[\]]+$', '', normalized)
        
        # Handle escaped quotes
        normalized = normalized.replace('""', '"').replace("''", "'")
        
        return normalized

    def _is_valid_identifier(self, name: str) -> bool:
        """
        Check if a string is a valid database identifier.
        
        Args:
            name: Name to validate
            
        Returns:
            True if valid identifier
        """
        if not name or not name.strip():
            return False
            
        normalized = self._normalize_name(name)
        
        patterns = [
            r'^[a-z_][a-z0-9_]*$',           # Standard: table_name
            r'^[a-z][a-z0-9]*$',             # Simple: tablename
            r'^[a-z][a-z0-9_-]*[a-z0-9]$',   # With hyphens: table-name
            r'^[a-z0-9\u0080-\uFFFF _\-\[\]`"\']+$',  # Unicode, quoted, etc.
        ]
        
        return any(re.match(pattern, normalized) for pattern in patterns)

    def _find_token_delimiter(self, full_ids: List[int], start_pos: int, end_pos: int, 
                              delimiter: str) -> Tuple[int, bool]:
        """
        Find the position of a specific delimiter in a tokenized sequence.
        
        Args:
            full_ids: Token ID list
            start_pos: Starting position
            end_pos: Ending position
            delimiter: Delimiter to find
            
        Returns:
            Tuple of (position, found_delimiter)
        """
        pos = start_pos
        while pos < end_pos:
            piece = self._decode([full_ids[pos]])
            if delimiter in piece:
                return pos, True
            pos += 1
        return pos, False

    def _extract_text_until_delimiter(self, full_ids: List[int], start_pos: int, end_pos: int, 
                                     delimiter: str) -> Tuple[str, int, bool]:
        """
        Extract text from start_pos until delimiter is found.
        
        Args:
            full_ids: Token ID list
            start_pos: Starting position
            end_pos: Ending position
            delimiter: Delimiter to stop at
            
        Returns:
            Tuple of (extracted_text, end_position, found_delimiter)
        """
        pos = start_pos
        accumulated_text = []
        
        while pos < end_pos:
            token_text = self._decode([full_ids[pos]])
            
            # Check if delimiter is in this token
            if delimiter in token_text:
                # Split on delimiter and add the first part
                parts = token_text.split(delimiter, 1)
                accumulated_text.append(parts[0])
                return "".join(accumulated_text), pos, True
            
            accumulated_text.append(token_text)
            pos += 1
            
        # Delimiter not found
        return "".join(accumulated_text), pos, False

    def _scan_tokens_until_delimiter(self, full_ids: List[int], start_pos: int, end_pos: int, 
                                    delimiter: str, max_tokens: int = 50) -> Tuple[List[int], int, bool]:
        """
        Scan tokens from start_pos until delimiter is found or max_tokens is reached.
        
        Args:
            full_ids: Token ID list
            start_pos: Starting position
            end_pos: Ending position
            delimiter: Delimiter to stop at
            max_tokens: Maximum number of tokens to scan
            
        Returns:
            Tuple of (collected_tokens, end_position, found_delimiter)
        """
        pos = start_pos
        accumulated_tokens = []
        tokens_scanned = 0
        
        while pos < end_pos and tokens_scanned < max_tokens:
            token_id = full_ids[pos]
            token_text = self._decode([token_id])
            
            # Check if delimiter is in this token
            if delimiter in token_text:
                accumulated_tokens.append(token_id)
                return accumulated_tokens, pos, True
            
            accumulated_tokens.append(token_id)
            pos += 1
            tokens_scanned += 1
            
        # Delimiter not found within max_tokens
        if pos >= end_pos:
            logger.debug(f"Scan reached end of range without finding delimiter '{delimiter}'")
        elif tokens_scanned >= max_tokens:
            logger.debug(f"Scan reached max token limit ({max_tokens}) without finding delimiter '{delimiter}'")
            
        return accumulated_tokens, pos, False

    def _extract_fk_references(self, payload: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Extract FK reference information from payload.
        
        Args:
            payload: FK payload string (e.g. "col_name -> ref_table.ref_col")
            
        Returns:
            Tuple of (ref_table, ref_column, column_name)
        """
        ref_tbl, ref_col = None, None
        col_part = payload.strip()
        
        # Handle various FK formats:
        # "col_name -> ref_table.ref_col"
        # "col_name->ref_table.ref_col" 
        # "col_name -> ref_table . ref_col"
        if "->" in payload:
            parts = payload.split("->", 1)
            col_part = parts[0].strip()
            ref_part = parts[1].strip()
            
            # Handle space around dot: "table . column" -> "table.column"
            ref_part = re.sub(r'\s*\.\s*', '.', ref_part)
            
            if "." in ref_part:
                ref_parts = ref_part.split(".", 1)
                ref_tbl = self._normalize_name(ref_parts[0])
                ref_col = self._normalize_name(ref_parts[1])
                
                # Validate reference names
                if not self._is_valid_identifier(ref_tbl):
                    logger.warning(f"Invalid FK reference table name: '{ref_parts[0]}'")
                    ref_tbl = None
                if not self._is_valid_identifier(ref_col):
                    logger.warning(f"Invalid FK reference column name: '{ref_parts[1]}'")
                    ref_col = None
        
        # Now extract just the column part (before type if present)
        if ":" in col_part:
            col_part = col_part.split(":", 1)[0].strip()
            
        col_part = self._normalize_name(col_part)
        return ref_tbl, ref_col, col_part

    def validate_schema_format(self, full_ids: List[int]) -> bool:
        """
        Validate schema format before parsing.
        
        Args:
            full_ids: List of token IDs
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if schema tokens exist
            if self.tok_id['SCHEMA_START'] not in full_ids or self.tok_id['SCHEMA_END'] not in full_ids:
                logger.error("Missing SCHEMA_START or SCHEMA_END tokens in input")
                return False
                
            # Validate schema region
            try:
                s_start = full_ids.index(self.tok_id['SCHEMA_START'])
                s_end = full_ids.index(self.tok_id['SCHEMA_END'])
                
                if s_start >= s_end:
                    logger.error(f"Invalid schema region: start={s_start}, end={s_end}")
                    return False
                    
                # Check for balanced special token pairs in schema region
                schema_region = full_ids[s_start:s_end+1]
                
                # Check PK token pairs
                pk_starts = schema_region.count(self.tok_id['PK_START'])
                pk_ends = schema_region.count(self.tok_id['PK_END'])
                if pk_starts != pk_ends:
                    logger.error(f"Unbalanced PK tags: {pk_starts} starts, {pk_ends} ends")
                    return False
                    
                # Check FK token pairs
                fk_starts = schema_region.count(self.tok_id['FK_START'])
                fk_ends = schema_region.count(self.tok_id['FK_END'])
                if fk_starts != fk_ends:
                    logger.error(f"Unbalanced FK tags: {fk_starts} starts, {fk_ends} ends")
                    return False
                
                return True
                
            except ValueError:
                logger.error("Invalid schema structure: could not find schema tokens")
                return False
                
        except Exception as e:
            logger.error(f"Error validating schema format: {e}")
            return False

    def _extract_table_definition(self, full_ids: List[int], start_pos: int, end_pos: int) -> Tuple[str, int, bool]:
        """
        Extract a table definition from tokenized schema.
        
        Args:
            full_ids: Token ID list
            start_pos: Starting position
            end_pos: Ending position
            
        Returns:
            Tuple of (table_name, end_position, success)
        """
        # First extract the table name (until opening parenthesis)
        table_name_text, table_name_end_pos, found_paren = self._extract_text_until_delimiter(
            full_ids, start_pos, end_pos, "("
        )
        
        if not found_paren:
            return "", start_pos, False
            
        # Normalize and validate table name
        table_name = self._normalize_name(table_name_text)
        if not self._is_valid_identifier(table_name):
            logger.warning(f"Invalid table name: '{table_name_text}' at position {start_pos}")
            return "", start_pos, False
            
        return table_name, table_name_end_pos, True

    def _find_matching_parenthesis(self, full_ids: List[int], start_pos: int, end_pos: int) -> Tuple[int, bool]:
        """
        Find matching closing parenthesis for a table definition.
        
        Args:
            full_ids: Token ID list
            start_pos: Start position (after opening parenthesis)
            end_pos: End position
            
        Returns:
            Tuple of (position_of_closing_paren, found)
        """
        pos = start_pos
        level = 1  # Start with level 1 as we're already after the opening parenthesis
        
        while pos < end_pos:
            token_text = self._decode([full_ids[pos]])
            
            if "(" in token_text:
                level += 1
            if ")" in token_text:
                level -= 1
                if level == 0:
                    return pos, True
            pos += 1
            
        return pos, False

    def _split_column_definitions(self, full_ids: List[int], start_pos: int, end_pos: int) -> Iterator[Tuple[int, int]]:
        """
        Split column definitions by commas, respecting PK and FK tags.
        
        Args:
            full_ids: Token ID list
            start_pos: Start position (inside table parentheses)
            end_pos: End position
            
        Returns:
            Iterator of (col_start, col_end) pairs
        """
        pos = start_pos
        col_start = pos
        in_special_tag = False
        special_tag_level = 0
        
        while pos < end_pos:
            token_id = full_ids[pos]
            
            # Keep track of PK/FK tags to avoid splitting on commas inside them
            if token_id == self.tok_id['PK_START'] or token_id == self.tok_id['FK_START']:
                in_special_tag = True
                special_tag_level += 1
            elif token_id == self.tok_id['PK_END'] or token_id == self.tok_id['FK_END']:
                special_tag_level -= 1
                if special_tag_level == 0:
                    in_special_tag = False
            
            token_text = self._decode([token_id])
            
            # Only split on commas outside of special tags
            if "," in token_text and not in_special_tag:
                # Column spans from col_start to pos (inclusive)
                yield col_start, pos
                col_start = pos + 1  # Next column starts after the comma
            
            pos += 1
        
        # Don't forget the last column if we have one
        if col_start < pos:
            yield col_start, pos - 1

    def _parse_pk_column(self, full_ids: List[int], start_pos: int, end_pos: int) -> Tuple[str, int, bool]:
        """
        Parse a primary key column definition.
        
        Args:
            full_ids: Token ID list
            start_pos: Start position (after PK_START)
            end_pos: End position (before PK_END)
            
        Returns:
            Tuple of (column_name, end_position, success)
        """
        col_tokens = full_ids[start_pos:end_pos]
        if not col_tokens:
            return "", end_pos, False
            
        # Extract column name (possibly with type)
        col_text = self._decode(col_tokens)
        
        # If there's a type specifier, extract just the column name
        if ":" in col_text:
            col_name = col_text.split(":", 1)[0].strip()
        else:
            col_name = col_text.strip()
            
        col_name = self._normalize_name(col_name)
        
        if not self._is_valid_identifier(col_name):
            logger.warning(f"Invalid PK column name: '{col_text}' at positions {start_pos}-{end_pos}")
            return "", end_pos, False
            
        return col_name, end_pos, True

    def _parse_fk_column(self, full_ids: List[int], start_pos: int, end_pos: int) -> Tuple[str, Tuple[str, str], int, bool]:
        """
        Parse a foreign key column definition.
        
        Args:
            full_ids: Token ID list
            start_pos: Start position (after FK_START)
            end_pos: End position (before FK_END)
            
        Returns:
            Tuple of (column_name, (ref_table, ref_column), end_position, success)
        """
        fk_tokens = full_ids[start_pos:end_pos]
        if not fk_tokens:
            return "", (None, None), end_pos, False
            
        fk_text = self._decode(fk_tokens)
        ref_table, ref_column, col_name = self._extract_fk_references(fk_text)
        
        if not self._is_valid_identifier(col_name):
            logger.warning(f"Invalid FK column name: '{fk_text}' at positions {start_pos}-{end_pos}")
            return "", (None, None), end_pos, False
            
        if not ref_table or not ref_column:
            logger.warning(f"Invalid FK reference: '{fk_text}' at positions {start_pos}-{end_pos}")
            return col_name, (None, None), end_pos, False
            
        return col_name, (ref_table, ref_column), end_pos, True

    def _parse_regular_column(self, full_ids: List[int], start_pos: int, end_pos: int) -> Tuple[str, int, bool]:
        """
        Parse a regular column definition.
        
        Args:
            full_ids: Token ID list
            start_pos: Start position
            end_pos: End position
            
        Returns:
            Tuple of (column_name, end_position, success)
        """
        col_tokens = full_ids[start_pos:end_pos+1]
        if not col_tokens:
            return "", end_pos, False
            
        col_text = self._decode(col_tokens).strip()
        
        # Extract column name (before type if present)
        if ":" in col_text:
            col_name = col_text.split(":", 1)[0].strip()
        else:
            col_name = col_text
            
        col_name = self._normalize_name(col_name)
        
        if not self._is_valid_identifier(col_name):
            logger.warning(f"Invalid column name: '{col_text}' at positions {start_pos}-{end_pos}")
            return "", end_pos, False
            
        return col_name, end_pos, True

    def parse_schema_tokens(self, full_ids: List[int]) -> List[SchemaToken]:
        """
        Parse schema tokens from input sequence with robust multi-token handling.

        Args:
            full_ids: List of token IDs

        Returns:
            List of SchemaToken objects
        """
        schema_tokens: List[SchemaToken] = []
        
        try:
            if not self.validate_schema_format(full_ids):
                logger.warning("Invalid schema format, returning empty schema tokens list")
                return schema_tokens
                
            # Extract schema region
            s_start = full_ids.index(self.tok_id['SCHEMA_START']) + 1
            s_end = full_ids.index(self.tok_id['SCHEMA_END'])
            
            logger.debug(f"Schema region spans from index {s_start} to {s_end}")
            
            # Decode and log schema region for debugging
            try:
                detok_schema = self._decode(full_ids[s_start:s_end])
                logger.debug(f"Detokenized schema: '{detok_schema}'")
            except Exception as e:
                logger.error(f"Could not detokenize schema region: {e}")
            
            # Process the schema region token by token, looking for table definitions
            pos = s_start
            current_table = None
            
            while pos < s_end:
                token_text = self._decode([full_ids[pos]])
                
                # Look for potential table definitions (contain table_name()
                if "(" in token_text or (pos + 1 < s_end and "(" in self._decode([full_ids[pos + 1]])):
                    # Try to extract table name at this position
                    table_name, table_end_pos, table_found = self._extract_table_definition(
                        full_ids, pos, s_end
                    )
                    
                    if table_found:
                        # Record the table token
                        table_token = SchemaToken(
                            span_start=pos,
                            span_end=table_end_pos,
                            token_type='table',
                            table_name=table_name
                        )
                        schema_tokens.append(table_token)
                        logger.debug(f"Found table: '{table_name}' at positions {pos}-{table_end_pos}")
                        
                        current_table = table_name
                        
                        # Now find the closing parenthesis for this table definition
                        table_content_start = table_end_pos + 1
                        closing_pos, found_closing = self._find_matching_parenthesis(
                            full_ids, table_content_start, s_end
                        )
                        
                        if found_closing:
                            # Process each column definition within the table
                            for col_start, col_end in self._split_column_definitions(
                                    full_ids, table_content_start, closing_pos):
                                
                                # Check if this column is a PK
                                if full_ids[col_start] == self.tok_id['PK_START']:
                                    # Find the matching PK_END
                                    pk_start = col_start
                                    pk_content_start = pk_start + 1
                                    pk_end = col_end
                                    
                                    # Search for PK_END
                                    for i in range(pk_content_start, col_end + 1):
                                        if full_ids[i] == self.tok_id['PK_END']:
                                            pk_end = i
                                            break
                                            
                                    if pk_end > pk_start:
                                        # Parse PK column
                                        col_name, _, success = self._parse_pk_column(
                                            full_ids, pk_content_start, pk_end
                                        )
                                        
                                        if success:
                                            pk_token = SchemaToken(
                                                span_start=pk_start,
                                                span_end=pk_end,
                                                token_type='pk',
                                                table_name=current_table,
                                                column_name=col_name
                                            )
                                            schema_tokens.append(pk_token)
                                            logger.debug(f"Found PK: '{col_name}' in table '{current_table}' at positions {pk_start}-{pk_end}")
                                
                                # Check if this column is an FK
                                elif full_ids[col_start] == self.tok_id['FK_START']:
                                    # Find the matching FK_END
                                    fk_start = col_start
                                    fk_content_start = fk_start + 1
                                    fk_end = col_end
                                    
                                    # Search for FK_END
                                    for i in range(fk_content_start, col_end + 1):
                                        if full_ids[i] == self.tok_id['FK_END']:
                                            fk_end = i
                                            break
                                            
                                    if fk_end > fk_start:
                                        # Parse FK column
                                        col_name, references, _, success = self._parse_fk_column(
                                            full_ids, fk_content_start, fk_end
                                        )
                                        
                                        if success:
                                            fk_token = SchemaToken(
                                                span_start=fk_start,
                                                span_end=fk_end,
                                                token_type='fk',
                                                table_name=current_table,
                                                column_name=col_name,
                                                references=references
                                            )
                                            schema_tokens.append(fk_token)
                                            logger.debug(f"Found FK: '{col_name}' in table '{current_table}' with reference to '{references[0]}.{references[1]}' at positions {fk_start}-{fk_end}")
                                
                                # Regular column
                                else:
                                    col_name, _, success = self._parse_regular_column(
                                        full_ids, col_start, col_end
                                    )
                                    
                                    if success:
                                        col_token = SchemaToken(
                                            span_start=col_start,
                                            span_end=col_end,
                                            token_type='column',
                                            table_name=current_table,
                                            column_name=col_name
                                        )
                                        schema_tokens.append(col_token)
                                        logger.debug(f"Found column: '{col_name}' in table '{current_table}' at positions {col_start}-{col_end}")
                            
                            # Move position to after the closing parenthesis
                            pos = closing_pos + 1
                            continue
                
                # If we didn't continue from within the table processing logic, increment position
                pos += 1
            
            # Log parsing summary
            token_types = {}
            for token in schema_tokens:
                token_types[token.token_type] = token_types.get(token.token_type, 0) + 1
                
            logger.info(f"Schema parsing summary: {token_types}")
            
            return schema_tokens
            
        except Exception as e:
            logger.error(f"Error parsing schema tokens: {e}", exc_info=True)
            return []

    def build_relation_matrix(self,
                            full_ids: List[int],
                            schema_tokens: Optional[List[SchemaToken]] = None,
                            max_seq_len_for_matrix: Optional[int] = None) -> torch.Tensor:
        """
        Build relation matrix from schema tokens.
        
        Args:
            full_ids: List of token IDs
            schema_tokens: Optional list of SchemaToken objects. If None, will be parsed.
            max_seq_len_for_matrix: Optional maximum sequence length to enforce for the matrix.
                                    If full_ids is longer, it will be truncated.
                                    If None, uses the length of full_ids.
            
        Returns:
            Relation matrix tensor of shape (seq_len, seq_len)
        """
        try:
            current_full_ids = list(full_ids)

            if max_seq_len_for_matrix is not None and len(current_full_ids) > max_seq_len_for_matrix:
                logger.warning(f"RelationMatrixBuilder: input full_ids (len {len(current_full_ids)}) exceeds max_seq_len_for_matrix ({max_seq_len_for_matrix}). Truncating full_ids.")
                current_full_ids = current_full_ids[:max_seq_len_for_matrix]
            
            seq_len = len(current_full_ids)
            
            rel = torch.zeros((seq_len, seq_len), dtype=torch.long)

            # Parse schema tokens if not provided, using the (potentially truncated) current_full_ids
            if schema_tokens is None:
                schema_tokens = self.parse_schema_tokens(current_full_ids)
                
            if not schema_tokens:
                logger.warning("No schema tokens to build relation matrix, returning zero matrix")
                return rel

            # Helper to set relation across token spans
            def set_rel(span_a: Tuple[int, int],
                        span_b: Tuple[int, int],
                        rel_id: int):
                try:
                    a0, a1 = span_a
                    b0, b1 = span_b
                    if a0 < 0 or a1 >= seq_len or b0 < 0 or b1 >= seq_len:
                        logger.warning(f"Invalid span indices: {span_a}, {span_b}, sequence length: {seq_len}")
                        return
                    rel[a0:a1+1, b0:b1+1] = rel_id
                except Exception as e:
                    logger.error(f"Error setting relation: {e}")

            # Compute relations
            for tok_i in schema_tokens:
                for tok_j in schema_tokens:
                    try:
                        # SAME TABLE
                        if (tok_i.table_name and tok_j.table_name
                                and tok_i.table_name == tok_j.table_name):
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['same_table'])

                        # PK ↔ FK
                        if tok_i.token_type == 'pk' and tok_j.token_type == 'fk' \
                           and tok_j.references and tok_i.table_name and tok_i.column_name \
                           and tok_i.table_name == tok_j.references[0] \
                           and tok_i.column_name == tok_j.references[1]:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['pk_fk'])
                        if tok_i.token_type == 'fk' and tok_j.token_type == 'pk' \
                           and tok_i.references and tok_j.table_name and tok_j.column_name \
                           and tok_j.table_name == tok_i.references[0] \
                           and tok_j.column_name == tok_i.references[1]:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['pk_fk'])

                        # TABLE–COLUMN
                        if tok_i.token_type == 'table' and tok_j.token_type in ['column', 'pk', 'fk'] \
                           and tok_i.table_name == tok_j.table_name:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['table_column'])
                        if tok_j.token_type == 'table' and tok_i.token_type in ['column', 'pk', 'fk'] \
                           and tok_i.table_name == tok_j.table_name:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['table_column'])

                        # SAME COLUMN NAME
                        if tok_i.token_type in ['column', 'pk', 'fk'] and tok_j.token_type in ['column', 'pk', 'fk'] \
                           and tok_i.column_name and tok_j.column_name \
                           and tok_i.column_name == tok_j.column_name \
                           and tok_i.table_name != tok_j.table_name:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['same_column'])
                    except Exception as e:
                        logger.error(f"Error computing relation between tokens: {e}")
                        continue

            # Validate relation matrix
            non_zero = (rel > 0).sum().item()
            if non_zero == 0:
                logger.warning("Relation matrix contains no non-zero entries")
            else:
                logger.debug(f"Relation matrix contains {non_zero} non-zero entries")

            return rel

        except Exception as e:
            logger.error(f"Error building relation matrix: {e}")
            return torch.zeros((len(full_ids), len(full_ids)), dtype=torch.long)
            
    def build_matrix(self, schema: str, max_seq_len_for_matrix: Optional[int] = None) -> torch.Tensor:
        """
        Convenience method to build relation matrix from schema string.
        
        Args:
            schema: Schema string
            max_seq_len_for_matrix: Optional maximum sequence length for the matrix.
            
        Returns:
            Relation matrix tensor
        """
        try:
            # Add schema tags if not present
            if not schema.startswith("<SCHEMA>"):
                schema = f"<SCHEMA> {schema} </SCHEMA>"
                
            # Tokenize schema
            tokens = self.sp.encode(schema)
            
            # Build relation matrix
            return self.build_relation_matrix(tokens, max_seq_len_for_matrix=max_seq_len_for_matrix)
            
        except Exception as e:
            logger.error(f"Error building matrix from schema string: {e}")
            # Return empty matrix as fallback
            size = max_seq_len_for_matrix if max_seq_len_for_matrix else 1
            return torch.zeros((size, size), dtype=torch.long)
            
    def _handle_consecutive_delimiters(self, full_ids: List[int], pos: int, s_end: int, delimiter: str) -> Tuple[int, bool]:
        """
        Handle cases where delimiters appear multiple times consecutively.
        
        Args:
            full_ids: Token ID list
            pos: Current position
            s_end: Schema end position
            delimiter: Delimiter to check for
            
        Returns:
            Tuple of (next_valid_position, found_valid_content)
        """
        consecutive_delims = 0
        j = pos
        
        while j < s_end:
            part = self._decode([full_ids[j]])
            if part.strip() == delimiter:
                consecutive_delims += 1
                j += 1
                if consecutive_delims > 3:  # Prevent infinite loops
                    logger.warning(f"Too many consecutive '{delimiter}' delimiters at position {pos}")
                    return j, False
            else:
                break
                
        return j, consecutive_delims > 0

    def _validate_parsed_token(self, token: SchemaToken) -> bool:
        """
        Validate a parsed schema token for consistency.
        
        Args:
            token: SchemaToken to validate
            
        Returns:
            True if valid
        """
        try:
            # Basic validation
            if token.span_start < 0 or token.span_end < token.span_start:
                return False
                
            # Type-specific validation
            if token.token_type == 'table':
                return bool(token.table_name and self._is_valid_identifier(token.table_name))
            elif token.token_type in ['column', 'pk']:
                return bool(token.column_name and self._is_valid_identifier(token.column_name))
            elif token.token_type == 'fk':
                valid_col = bool(token.column_name and self._is_valid_identifier(token.column_name))
                if token.references:
                    ref_tbl, ref_col = token.references
                    valid_ref = bool(ref_tbl and ref_col and 
                                   self._is_valid_identifier(ref_tbl) and 
                                   self._is_valid_identifier(ref_col))
                    return valid_col and valid_ref
                return valid_col
                
            return True
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False
