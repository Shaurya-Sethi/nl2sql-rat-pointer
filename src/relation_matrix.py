import torch
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import sentencepiece as spm

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
                 sp_model_path: str,
                 special_tokens: Dict[str, str],   # pieces, e.g. '<PK>'
                 num_relations: int = 5):
        """
        Initialize relation matrix builder.
        
        Args:
            sp_model_path: Path to sentencepiece model
            special_tokens: Dictionary of special token strings
            num_relations: Number of relation types
        """
        try:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)

            # Validate special tokens
            required_tokens = {
                'SCHEMA_START', 'SCHEMA_END',
                'PK_START', 'PK_END',
                'FK_START', 'FK_END'
            }
            missing_tokens = required_tokens - set(special_tokens.keys())
            if missing_tokens:
                raise ValueError(f"Missing required special tokens: {missing_tokens}")

            # Convert special-piece strings to their IDs
            self.tok_id = {}
            for k, v in special_tokens.items():
                try:
                    self.tok_id[k] = self.sp.piece_to_id(v)
                except Exception as e:
                    raise ValueError(f"Invalid special token '{k}': {e}")

            self.num_relations = num_relations
            
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
            logger.error(f"Error decoding tokens: {e}")
            return ""

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

            # Find schema region
            try:
                s_start = full_ids.index(self.tok_id['SCHEMA_START']) + 1
                s_end = full_ids.index(self.tok_id['SCHEMA_END'])
            except ValueError:
                logger.warning("No schema region found")
                return schema_tokens

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
                            logger.warning("Unclosed PK tag")
                            break
                        col_name = self._decode(full_ids[i+1:j])
                        schema_tokens.append(SchemaToken(
                            span_start=i, span_end=j,
                            token_type='pk',
                            table_name=current_table,
                            column_name=col_name,
                        ))
                        i = j + 1
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing PK: {e}")
                        i += 1
                        continue

                # Parse FK
                if tok == self.tok_id['FK_START']:
                    try:
                        j = i + 1
                        while j < s_end and full_ids[j] != self.tok_id['FK_END']:
                            j += 1
                        if j >= s_end:
                            logger.warning("Unclosed FK tag")
                            break
                        payload = self._decode(full_ids[i+1:j])
                        if "->" in payload:
                            col_part, ref_part = [p.strip() for p in payload.split("->", 1)]
                            if "." in ref_part:
                                ref_tbl, ref_col = [p.strip() for p in ref_part.split(".", 1)]
                            else:
                                ref_tbl, ref_col = None, None
                        else:
                            col_part, ref_tbl, ref_col = payload, None, None
                        schema_tokens.append(SchemaToken(
                            span_start=i, span_end=j,
                            token_type='fk',
                            table_name=current_table,
                            column_name=col_part,
                            references=(ref_tbl, ref_col) if ref_tbl else None
                        ))
                        i = j + 1
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing FK: {e}")
                        i += 1
                        continue

                # Parse table or column
                piece = self._decode([tok])
                if "(" in piece:
                    try:
                        table_name = piece.split("(")[0].strip()
                        current_table = table_name
                        schema_tokens.append(SchemaToken(
                            span_start=i, span_end=i,
                            token_type='table',
                            table_name=current_table
                        ))
                        i += 1
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing table: {e}")
                        i += 1
                        continue

                if ":" in piece:
                    try:
                        col_name = piece.split(":")[0].strip()
                        schema_tokens.append(SchemaToken(
                            span_start=i, span_end=i,
                            token_type='column',
                            table_name=current_table,
                            column_name=col_name
                        ))
                        i += 1
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing column: {e}")
                        i += 1
                        continue

                i += 1

            return schema_tokens

        except Exception as e:
            logger.error(f"Error parsing schema tokens: {e}")
            return []

    def build_relation_matrix(self,
                            full_ids: List[int],
                            schema_tokens: List[SchemaToken]) -> torch.Tensor:
        """
        Build relation matrix from schema tokens.
        
        Args:
            full_ids: List of token IDs
            schema_tokens: List of SchemaToken objects
            
        Returns:
            Relation matrix tensor of shape (seq_len, seq_len)
        """
        try:
            seq_len = len(full_ids)
            rel = torch.zeros((seq_len, seq_len), dtype=torch.long)

            # Helper to set relation across token spans
            def set_rel(span_a: Tuple[int, int],
                        span_b: Tuple[int, int],
                        rel_id: int):
                try:
                    a0, a1 = span_a
                    b0, b1 = span_b
                    if a0 < 0 or a1 >= seq_len or b0 < 0 or b1 >= seq_len:
                        logger.warning(f"Invalid span indices: {span_a}, {span_b}")
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
                           and tok_j.references \
                           and tok_i.table_name == tok_j.references[0] \
                           and tok_i.column_name == tok_j.references[1]:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['pk_fk'])
                        if tok_i.token_type == 'fk' and tok_j.token_type == 'pk' \
                           and tok_i.references \
                           and tok_j.table_name == tok_i.references[0] \
                           and tok_j.column_name == tok_i.references[1]:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['pk_fk'])

                        # TABLE–COLUMN
                        if tok_i.token_type == 'table' and tok_j.token_type == 'column' \
                           and tok_i.table_name == tok_j.table_name:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['table_column'])
                        if tok_j.token_type == 'table' and tok_i.token_type == 'column' \
                           and tok_i.table_name == tok_j.table_name:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['table_column'])

                        # SAME COLUMN NAME
                        if tok_i.token_type == 'column' and tok_j.token_type == 'column' \
                           and tok_i.column_name == tok_j.column_name \
                           and tok_i.table_name != tok_j.table_name:
                            set_rel((tok_i.span_start, tok_i.span_end),
                                    (tok_j.span_start, tok_j.span_end),
                                    self._REL_MAP['same_column'])
                    except Exception as e:
                        logger.error(f"Error computing relation between tokens: {e}")
                        continue

            return rel

        except Exception as e:
            logger.error(f"Error building relation matrix: {e}")
            return torch.zeros((seq_len, seq_len), dtype=torch.long)