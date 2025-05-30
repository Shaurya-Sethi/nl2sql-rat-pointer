"""
formatter.py – Schema → single-line text for NL2SQL (composite PK grouping)

Key change
----------
• For any table with ≥1 PK columns, they are emitted **once** inside a single
  <PK> … </PK> wrapper, comma-separated:
      table1(<PK> a:int, b:int </PK>, col3:text, …)

All other behaviour (FK wrapper, order, SPECIAL_TOKENS) is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

# src/ added for NL2SQLTokenizer import when this script is run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer import NL2SQLTokenizer  # noqa: E402  pylint: disable=wrong-import-position

SPECIAL_TOKENS = {
    "SCHEMA_START": "<SCHEMA>",
    "SCHEMA_END": "</SCHEMA>",
    "PK_START": "<PK>",
    "PK_END": "</PK>",
    "FK_START": "<FK>",
    "FK_END": "</FK>",
    "NL_START": "<NL>",
    "NL_END": "</NL>",
    "COT_START": "<COT>",
    "COT_END": "</COT>",
    "SQL_START": "<SQL>",
    "SQL_END": "</SQL>",
    "EXT_START": "<EXT>",
    "EXT_END": "</EXT>",
}


# ────────────────────────────────────────────────────────────────────────────
def _format_table(table: Dict, fk_map: Dict[str, set]) -> str:
    """Return single-table string with grouped <PK> wrapper if needed."""
    tname = table["name"]
    pk_cols = [c for c in table["columns"] if c.get("pk")]
    fk_cols = fk_map.get(tname, set())

    col_chunks: List[str] = []

    # -- 1. Composite-PK wrapper (even for single-PK) -----------------------
    if pk_cols:
        pk_inner = ", ".join(f"{c['name']}:{c['type']}" for c in pk_cols)
        col_chunks.append(f"{SPECIAL_TOKENS['PK_START']} {pk_inner} {SPECIAL_TOKENS['PK_END']}")

    # -- 2. Non-PK columns ---------------------------------------------------
    for col in table["columns"]:
        if col.get("pk"):
            continue  # already handled in the grouped wrapper

        cname, ctype = col["name"], col["type"]
        if cname in fk_cols:
            col_chunks.append(f"{SPECIAL_TOKENS['FK_START']} {cname}:{ctype} {SPECIAL_TOKENS['FK_END']}")
        else:
            col_chunks.append(f"{cname}:{ctype}")

    return f"{tname}({','.join(col_chunks)})"


def format_schema(schema: Dict) -> str:
    """
    Convert extractor schema dict into required single-line format
    (<SCHEMA> … </SCHEMA>) with composite-PK grouping.
    """
    # Build FK lookup first
    fk_map: Dict[str, set] = {}
    for fk in schema.get("foreign_keys", []):
        fk_map.setdefault(fk["from_table"], set()).add(fk["from_column"])

    tables_txt = " ".join(_format_table(t, fk_map) for t in schema["tables"])
    return f"{SPECIAL_TOKENS['SCHEMA_START']} {tables_txt} {SPECIAL_TOKENS['SCHEMA_END']}"


# Optional helper
def tokenize_schema(schema_text: str, sp_model_path: str, special_tokens: Dict) -> List[int]:
    tokenizer = NL2SQLTokenizer(sp_model_path, special_tokens)
    return tokenizer.encode(schema_text, add_special_tokens=False)


# ── CLI test ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--schema-json", required=True)
    p.add_argument("--sp-model")
    p.add_argument("--ids", action="store_true")
    args = p.parse_args()

    with open(args.schema_json, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)

    text = format_schema(schema_dict)

    if args.sp_model and args.ids:
        ids = tokenize_schema(text, args.sp_model, SPECIAL_TOKENS)
        print(" ".join(map(str, ids)))
    else:
        print(text)