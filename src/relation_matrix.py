# -*- coding: utf-8 -*-
"""
Refactored Relation Matrix Builder
=================================
This version focuses on *clarity* and *robustness*:

* **Single‑responsibility helpers** – each small method does one thing.
* **Early schema extraction** – the full input is sliced to the
  `<SCHEMA> … </SCHEMA>` span before any parsing. Internal logic never
  sees NL / COT / SQL tokens again.
* **Token‑driven parsing** – we walk the token list once, recognising
  tables & columns by the presence of `(` and special PK/FK wrappers.
* **Leaner dependency surface** – no regex over token streams, no
  multi‑stage caches; decoding is minimal & memoised.
* **Guaranteed relation count** – `_REL_MAP` is a single source of truth;
  constructor asserts `num_relations ≥ len(_REL_MAP)`.

compatible with the previous class: `build_relation_matrix`
returns a `torch.LongTensor (seq_len × seq_len)` with the same relation IDs.

>>> rmb = RelationMatrixBuilder(tokenizer)
>>> rel = rmb.build_relation_matrix(full_ids)  # works like before

If we hit an assertion or error, enable DEBUG logging – the walker prints a
step‑by‑step trace of its decisions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

DEBUG_VERBOSE = False  # Set True to enable detailed relation matrix build logging

def debug_log(msg, logger=None, *args):
    """Helper function for debug logging that respects DEBUG_VERBOSE flag"""
    if DEBUG_VERBOSE:
        if logger:
            logger.debug(msg, *args)
        else:
            print(msg)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # flip to DEBUG while debugging

__all__ = [
    "SchemaToken",
    "RelationMatrixBuilder",
]


# ---------------------------------------------------------------------------
# Dataclass ‑‑ schema token metadata
# ---------------------------------------------------------------------------
@dataclass
class SchemaToken:
    span_start: int  # inclusive
    span_end: int    # inclusive
    token_type: str  # "table" | "column" | "pk" | "fk"
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    references: Optional[Tuple[str, str]] = None  # (ref_table, ref_column)

    # --- helpers -----------------------------------------------------------
    def key(self) -> Tuple[str, str, str]:
        return self.token_type, self.table_name or "", self.column_name or ""

    # nice for debugging in logs
    def __repr__(self) -> str:  # pragma: no cover
        ref = (
            f" → {self.references[0]}.{self.references[1]}" if self.references else ""
        )
        return (
            f"<{self.token_type}:{self.table_name or ''}.{self.column_name or ''}{ref}"  # noqa:E501
            f" [{self.span_start}:{self.span_end}]>"
        )


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
class RelationMatrixBuilder:
    """Parse the schema segment and build an **NxN** relation matrix."""

    # relation‑type IDs – extend here if you add more relation kinds
    _REL_MAP: Dict[str, int] = {
        "no_relation": 0,
        "same_table": 1,
        "pk_fk": 2,
        "table_column": 3,
        "same_column": 4,
    }

    # ---------------------------------------------------------------------
    # Setup / helpers
    # ---------------------------------------------------------------------
    def __init__(self, tokenizer, num_relations: int = 5):
        self.sp = tokenizer.sp  # sentencepiece model (must expose .encode/.decode)
        self.tok_id = tokenizer.special_token_ids  # name → id

        # sanity – make sure we have our required specials
        required = {
            "SCHEMA_START",
            "SCHEMA_END",
            "PK_START",
            "PK_END",
            "FK_START",
            "FK_END",
        }
        missing = required - set(self.tok_id)
        if missing:
            raise ValueError(f"tokenizer.special_token_ids missing {missing}")

        if num_relations < len(self._REL_MAP):
            raise ValueError("num_relations must cover all relation kinds")
        self.num_relations = num_relations

        # micro cache for decode – avoid re‑decoding tiny tokens repeatedly
        self._decode_cache: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    def _decode(self, tid: int) -> str:
        if tid not in self._decode_cache:
            try:
                self._decode_cache[tid] = self.sp.decode([tid])
            except Exception:  # pragma: no cover
                self._decode_cache[tid] = ""
        return self._decode_cache[tid]

    @staticmethod
    def _norm(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip()).lower().strip("`\"[]")

    @staticmethod
    def _is_ident(name: str) -> bool:
        return bool(re.match(r"^[a-z_][a-z0-9_]*$", name))

    # ------------------------------------------------------------------
    # Phase 1 – slice out the schema       full_ids  →  schema_ids
    # ------------------------------------------------------------------
    def _extract_schema_segment(self, full_ids: List[int]) -> Tuple[List[int], int]:
        """Return the sub‑list *between* <SCHEMA> and </SCHEMA> and its start offset."""
        try:
            s = full_ids.index(self.tok_id["SCHEMA_START"]) + 1
            e = full_ids.index(self.tok_id["SCHEMA_END"])
        except ValueError:
            logger.error("<SCHEMA> or </SCHEMA> tokens missing – aborting parse")
            return [], 0 # Return 0 offset on error
        if s >= e:
            logger.error("Malformed schema segment – start ≥ end")
            return [], 0 # Return 0 offset on error
        # The offset is 's', the index of the first token *within* the schema segment,
        # relative to full_ids.
        return full_ids[s:e], s

    # ------------------------------------------------------------------
    # Phase 2 – walk schema tokens and materialise SchemaToken objects
    # ------------------------------------------------------------------
    def parse_schema_tokens(self, full_ids: List[int]) -> List[SchemaToken]:
        schema_ids, schema_start_offset = self._extract_schema_segment(full_ids)
        if not schema_ids:
            return []

        tokens: List[SchemaToken] = []
        pos = 0
        current_table: Optional[str] = None

        while pos < len(schema_ids):
            tid = schema_ids[pos]

            # --- PK -------------------------------------------------------
            if tid == self.tok_id["PK_START"]:
                pk_start = pos
                try:
                    pk_end = schema_ids.index(self.tok_id["PK_END"], pk_start + 1)
                except ValueError:
                    logger.warning("Unclosed <PK> tag – skipping")
                    break
                col_tokens = schema_ids[pk_start + 1 : pk_end]
                col_name = self._norm(self.sp.decode(col_tokens).split(":", 1)[0])
                if self._is_ident(col_name):
                    tokens.append(
                        SchemaToken(
                            span_start=pk_start + schema_start_offset,
                            span_end=pk_end + schema_start_offset,
                            token_type="pk",
                            table_name=current_table,
                            column_name=col_name,
                        )
                    )
                pos = pk_end + 1
                continue

            # --- FK -------------------------------------------------------
            if tid == self.tok_id["FK_START"]:
                fk_start = pos
                try:
                    fk_end = schema_ids.index(self.tok_id["FK_END"], fk_start + 1)
                except ValueError:
                    logger.warning("Unclosed <FK> tag – skipping")
                    break
                raw = self.sp.decode(schema_ids[fk_start + 1 : fk_end])
                parts = [p.strip() for p in raw.split("->", 1)]
                col_name = self._norm(parts[0].split(":", 1)[0])
                ref = (None, None)
                if len(parts) == 2 and "." in parts[1]:
                    ref_tbl, ref_col = parts[1].split(".", 1)
                    ref = (self._norm(ref_tbl), self._norm(ref_col))
                if self._is_ident(col_name):
                    tokens.append(
                        SchemaToken(
                            span_start=fk_start + schema_start_offset,
                            span_end=fk_end + schema_start_offset,
                            token_type="fk",
                            table_name=current_table,
                            column_name=col_name,
                            references=ref if all(ref) else None,
                        )
                    )
                pos = fk_end + 1
                continue

            # --- table or regular column -------------------------------
            tk_text = self._decode(tid)
            next_text = self._decode(schema_ids[pos + 1]) if pos + 1 < len(schema_ids) else ""

            # • Detect start of a table def: `table_name (` or `table_name(`
            if ("(" in next_text and self._is_ident(self._norm(tk_text))) or (
                "(" in tk_text and self._is_ident(self._norm(tk_text.split("(")[0]))
            ):
                # grab all tokens until matching ')' (depth 0)
                table_name = self._norm(tk_text.split("(")[0])
                table_start = pos
                depth = tk_text.count("(")
                j = pos + 1
                while j < len(schema_ids) and depth:
                    part = self._decode(schema_ids[j])
                    depth += part.count("(") - part.count(")")
                    j += 1
                table_end = j - 1
                tokens.append(
                    SchemaToken(
                        span_start=table_start + schema_start_offset,
                        span_end=table_end + schema_start_offset,
                        token_type="table",
                        table_name=table_name,
                    )
                )
                current_table = table_name
                pos = table_start + 1  # enter columns loop normally
                continue

            # • Regular column (outside PK/FK) – we mark each token span up to next comma/paren
            if current_table and self._is_ident(self._norm(tk_text)):
                col_start = pos
                j = pos + 1
                while j < len(schema_ids):
                    txt = self._decode(schema_ids[j])
                    if txt.strip() in {",", ")"}:
                        break
                    j += 1
                col_end = j - 1
                col_name = self._norm(tk_text.split(":", 1)[0])
                tokens.append(
                    SchemaToken(
                        span_start=col_start + schema_start_offset,
                        span_end=col_end + schema_start_offset,
                        token_type="column",
                        table_name=current_table,
                        column_name=col_name,
                    )
                )
                pos = j  # jump past column definition
                continue

            # Anything else – advance one token
            pos += 1

        debug_log("Parsed schema tokens: %s", logger, tokens)
        return tokens

    # ------------------------------------------------------------------
    # Phase 3 – build relation matrix
    # ------------------------------------------------------------------
    def build_relation_matrix(
        self,
        full_ids: List[int],
        schema_tokens: Optional[List[SchemaToken]] = None,
    ) -> torch.Tensor:
        if schema_tokens is None:
            schema_tokens = self.parse_schema_tokens(full_ids)
        seq_len = len(full_ids)
        rel = torch.zeros((seq_len, seq_len), dtype=torch.long)

        # IMPORTANT: The SchemaToken spans (i.k. span_start, i.span_end) are now GLOBAL,
        # meaning they are relative to the `full_ids` sequence.
        # This is crucial for correctly placing relations in the matrix when the
        # schema is preceded by other tokens (e.g., Natural Language).
        def set_span(a: Tuple[int, int], b: Tuple[int, int], rel_id: int):
            # Ensure spans are within bounds of the full relation matrix
            # This check is more critical now that spans are global.
            if not (0 <= a[0] < seq_len and 0 <= a[1] < seq_len and \
                    0 <= b[0] < seq_len and 0 <= b[1] < seq_len):
                logger.warning(f"Span out of bounds: a={a}, b={b}, seq_len={seq_len}. Skipping relation.")
                return
            rel[a[0] : a[1] + 1, b[0] : b[1] + 1] = rel_id

        for i in schema_tokens:
            for j in schema_tokens:
                # same table ------------------------------------------------
                if i.table_name and i.table_name == j.table_name:
                    set_span((i.span_start, i.span_end), (j.span_start, j.span_end), self._REL_MAP["same_table"])

                # pk ↔ fk ---------------------------------------------------
                if i.token_type == "pk" and j.token_type == "fk" and j.references == (
                    i.table_name,
                    i.column_name,
                ):
                    set_span((i.span_start, i.span_end), (j.span_start, j.span_end), self._REL_MAP["pk_fk"])
                if j.token_type == "pk" and i.token_type == "fk" and i.references == (
                    j.table_name,
                    j.column_name,
                ):
                    set_span((i.span_start, i.span_end), (j.span_start, j.span_end), self._REL_MAP["pk_fk"])

                # table ↔ column -------------------------------------------
                if i.token_type == "table" and j.token_type in {"column", "pk", "fk"} and i.table_name == j.table_name:
                    set_span((i.span_start, i.span_end), (j.span_start, j.span_end), self._REL_MAP["table_column"])
                if j.token_type == "table" and i.token_type in {"column", "pk", "fk"} and j.table_name == i.table_name:
                    set_span((i.span_start, i.span_end), (j.span_start, j.span_end), self._REL_MAP["table_column"])

                # same column name across tables ---------------------------
                if (
                    i.token_type in {"column", "pk", "fk"}
                    and j.token_type in {"column", "pk", "fk"}
                    and i.column_name
                    and i.column_name == j.column_name
                    and i.table_name != j.table_name
                ):
                    set_span((i.span_start, i.span_end), (j.span_start, j.span_end), self._REL_MAP["same_column"])

        nz = (rel > 0).sum().item()
        debug_log("Relation matrix built – %d non‑zero cells", logger, nz)
        return rel
