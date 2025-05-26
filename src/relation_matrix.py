import torch
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from tokenizer import NL2SQLTokenizer
import re  # Added for regex pattern validation

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

    def __init__(self,
                 tokenizer: NL2SQLTokenizer,
                 num_relations: int = 5):
        """
        Initialize relation matrix builder.
        
        Args:
            tokenizer: NL2SQLTokenizer instance
            num_relations: Number of relation types
        """
        try:
            self.sp = tokenizer.sp
            self.tok_id = tokenizer.special_token_ids

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
        Decode token IDs to string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        """
        try:
            return self.sp.decode(ids) if ids else ""
        except Exception as e:
            logger.error(f"Error decoding tokens {ids}: {e}")
            return ""

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

def parse_schema_tokens(self, full_ids: List[int]) -> List[SchemaToken]:
    """
    Parse schema tokens from input sequence.

    Args:
        full_ids: List of token IDs

    Returns:
        List of SchemaToken objects
    """
    try:
        schema_tokens: List[SchemaToken] = []

        # Validate schema format before parsing
        if not self.validate_schema_format(full_ids):
            logger.warning("Invalid schema format, returning empty schema tokens list")
            return schema_tokens

        # Find schema region
        try:
            s_start = full_ids.index(self.tok_id['SCHEMA_START']) + 1
            s_end = full_ids.index(self.tok_id['SCHEMA_END'])
        except ValueError:
            logger.warning("No schema region found")
            return schema_tokens

        schema_region_ids = full_ids[s_start:s_end]
        try:
            detok_schema_region = self._decode(schema_region_ids)
        except Exception as e:
            detok_schema_region = f"<decode error: {e}>"
        logger.warning(f"[SCHEMA_PARSE_DEBUG] Schema region token IDs: {schema_region_ids}")
        logger.warning(f"[SCHEMA_PARSE_DEBUG] Detokenized schema region: '{detok_schema_region}'")

        i = s_start
        current_table: Optional[str] = None

        while i < s_end:
            tok = full_ids[i]

            # Parse PK
            if tok == self.tok_id['PK_START']:
                try:
                    j = i + 1
                    while j < s_end and full_ids[j] != self.tok_id['PK_END']:
                        j += 1
                    if j >= s_end:
                        logger.warning(f"[SCHEMA_PARSE_DEBUG] Unclosed PK tag at position {i}, skipping. Schema segment: {full_ids[i:j+3]} | Detok: '{self._decode(full_ids[i:j+3])}'")
                        i += 1
                        continue
                    col_name = self._decode(full_ids[i+1:j])
                    if not col_name:
                        logger.warning(f"[SCHEMA_PARSE_DEBUG] Empty PK column name at position {i}, skipping. Token IDs: {full_ids[i:j]} | Detok: '{self._decode(full_ids[i:j])}'")
                        i = j + 1
                        continue
                    schema_tokens.append(SchemaToken(
                        span_start=i, span_end=j,
                        token_type='pk',
                        table_name=current_table,
                        column_name=col_name,
                    ))
                    i = j + 1
                    continue
                except Exception as e:
                    logger.error(f"Error parsing PK at position {i}: {e}")
                    i += 1
                    continue

            # Parse FK
            if tok == self.tok_id['FK_START']:
                try:
                    j = i + 1
                    while j < s_end and full_ids[j] != self.tok_id['FK_END']:
                        j += 1
                    if j >= s_end:
                        logger.warning(f"[SCHEMA_PARSE_DEBUG] Unclosed FK tag at position {i}, skipping. Schema segment: {full_ids[i:j+3]} | Detok: '{self._decode(full_ids[i:j+3])}'")
                        i += 1
                        continue
                    payload = self._decode(full_ids[i+1:j])
                    if not payload:
                        logger.warning(f"[SCHEMA_PARSE_DEBUG] Empty FK payload at position {i}, skipping. Token IDs: {full_ids[i:j]} | Detok: '{self._decode(full_ids[i:j])}'")
                        i = j + 1
                        continue
                    ref_tbl, ref_col = None, None
                    if "->" in payload:
                        col_part, ref_part = [p.strip() for p in payload.split("->", 1)]
                        if "." in ref_part:
                            ref_tbl, ref_col = [p.strip() for p in ref_part.split(".", 1)]
                    else:
                        col_part = payload
                    schema_tokens.append(SchemaToken(
                        span_start=i, span_end=j,
                        token_type='fk',
                        table_name=current_table,
                        column_name=col_part,
                        references=(ref_tbl, ref_col) if ref_tbl and ref_col else None
                    ))
                    i = j + 1
                    continue
                except Exception as e:
                    logger.error(f"Error parsing FK at position {i}: {e}")
                    i += 1
                    continue

            # PATCH: Robust multi-token table name extraction
            try:
                piece = self._decode([tok])
                # NEW TABLE LOGIC: gather tokens until one contains '('
                if "(" in piece or piece == "(":
                    t_start = i
                    table_tokens = []
                    j = i
                    while j < s_end:
                        part = self._decode([full_ids[j]])
                        table_tokens.append(part)
                        if "(" in part:
                            break
                        j += 1
                    table_name = "".join(table_tokens).split("(", 1)[0].strip()
                    if table_name:
                        current_table = table_name
                        schema_tokens.append(SchemaToken(
                            span_start=t_start, span_end=j,
                            token_type='table',
                            table_name=current_table
                        ))
                    else:
                        logger.warning(f"[SCHEMA_PARSE_DEBUG] Empty table name at position {t_start}-{j}, skipping. Token IDs: {full_ids[t_start:j+1]} | Detok: '{''.join(table_tokens)}'")
                    i = j + 1
                    continue
                # Column pattern: name followed by colon
                if ":" in piece:
                    col_name = piece.split(":", 1)[0].strip()
                    if not col_name:
                        logger.warning(f"[SCHEMA_PARSE_DEBUG] Empty column name at position {i}, skipping. Token ID: {tok} | Detok: '{piece}' | Schema region: {schema_region_ids}")
                        i += 1
                        continue
                    schema_tokens.append(SchemaToken(
                        span_start=i, span_end=i,
                        token_type='column',
                        table_name=current_table,
                        column_name=col_name
                    ))
                    i += 1
                    continue
            except Exception as e:
                logger.error(f"Error parsing token at position {i}: {e}")
                i += 1
                continue

            i += 1

        # Validate parsed schema tokens
        if not schema_tokens:
            logger.warning("No schema tokens parsed from valid schema region")

        # Log schema parsing summary
        token_types = {}
        for token in schema_tokens:
            token_types[token.token_type] = token_types.get(token.token_type, 0) + 1
        logger.info(f"Schema parsing summary: {token_types}")

        return schema_tokens

    except Exception as e:
        logger.error(f"Error parsing schema tokens: {e}")
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
                        if tok_i.token_type == 'column' and tok_j.token_type == 'column' \
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
                logger.info(f"Relation matrix contains {non_zero} non-zero entries")

            return rel

        except Exception as e:
            logger.error(f"Error building relation matrix: {e}")
            return torch.zeros((seq_len, seq_len), dtype=torch.long)
            
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
            # Validate schema string
            if not schema:
                logger.error("Empty schema string")
                return torch.zeros((1, 1), dtype=torch.long)
                
            # Encode schema
            schema_ids = self.sp.encode(schema)
            if not schema_ids:
                logger.error("Schema encoding produced empty ID list")
                return torch.zeros((1, 1), dtype=torch.long)
            
            # Check if encoded schema length exceeds the maximum if provided
            if max_seq_len_for_matrix is not None and len(schema_ids) > max_seq_len_for_matrix:
                logger.warning(f"RelationMatrixBuilder.build_matrix: Encoded schema (len {len(schema_ids)}) exceeds max_seq_len_for_matrix ({max_seq_len_for_matrix}). Truncating schema_ids.")
                schema_ids = schema_ids[:max_seq_len_for_matrix]
                
            # Build relation matrix
            return self.build_relation_matrix(schema_ids, max_seq_len_for_matrix=max_seq_len_for_matrix)
            
        except Exception as e:
            logger.error(f"Error in build_matrix: {e}")
            # Fallback based on original schema string length if schema_ids failed or max_seq_len is complex
            fallback_len = max_seq_len_for_matrix if max_seq_len_for_matrix is not None else (len(schema) if schema else 1)
            return torch.zeros((fallback_len, fallback_len), dtype=torch.long)