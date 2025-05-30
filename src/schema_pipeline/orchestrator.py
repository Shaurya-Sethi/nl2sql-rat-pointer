"""
orchestrator.py – RAG with dynamic threshold & safe fallback
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import formatter
from graph import build_schema_graph
from selector import build_table_embeddings, select_tables
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

SPECIAL = formatter.SPECIAL_TOKENS


class SchemaRAGOrchestrator:
    """
    End-to-end prompt builder.

    • Uses dynamic similarity threshold (selector.py).  
    • If selector prunes to **zero** tables, reverts to full schema.  
    """

    def __init__(
        self,
        tokenizer,
        schema: Dict,
        embed_model: SentenceTransformer | None = None,
        base_sim_threshold: float = 0.20,
        max_schema_tokens: int = 1528,
        max_question_tokens: int = 64,
    ):
        self.tok = tokenizer
        self.base_sim_threshold = base_sim_threshold
        self.max_schema_tokens = max_schema_tokens
        self.max_question_tokens = max_question_tokens

        # Cache
        self._schema = schema
        self._schema_text = formatter.format_schema(schema)
        self._schema_tokens: List[int] = self.tok.encode(self._schema_text, add_special_tokens=False)

        self._G = build_schema_graph(schema)
        self._embed = embed_model or SentenceTransformer("all-MiniLM-L6-v2")
        self._table_embs = build_table_embeddings(schema, self._embed)

    # --------------------------------------------------------------------- #
    def _encode_len(self, txt: str) -> int:
        try:
            return len(self.tok.encode(txt, add_special_tokens=False))
        except Exception:
            return len(txt.split()) * 2

    def _prune_schema(self, keep: List[str]) -> Tuple[str, List[int]]:
        ks = set(keep)
        pruned = {
            "tables": [t for t in self._schema["tables"] if t["name"] in ks],
            "foreign_keys": [
                fk
                for fk in self._schema.get("foreign_keys", [])
                if fk["from_table"] in ks and fk["to_table"] in ks
            ],
        }
        text = formatter.format_schema(pruned)
        return text, self.tok.encode(text, add_special_tokens=False)

    # --------------------------------------------------------------------- #
    def build_prompt(self, question: str) -> Tuple[str, List[int], Dict]:
        q_len = self._encode_len(question)
        s_len = len(self._schema_tokens)
        use_rag = s_len > self.max_schema_tokens or q_len > self.max_question_tokens

        if use_rag:
            keep_tables = select_tables(
                question,
                self._G,
                self._table_embs,
                self._embed,
                base_threshold=self.base_sim_threshold,
            )

            if keep_tables:  # normal RAG path
                schema_text, _ = self._prune_schema(keep_tables)
                rag_used = True
            else:  # fallback: selector returned none
                logger.warning("Selector returned 0 tables → reverting to full schema")
                schema_text, rag_used = self._schema_text, False
        else:
            schema_text, rag_used = self._schema_text, False

        prompt_text = f"{SPECIAL['NL_START']} {question} {SPECIAL['NL_END']} {schema_text}"
        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)

        info = {"rag_used": rag_used, "prompt_tokens": len(prompt_ids)}
        logger.debug("Prompt | rag=%s | tokens=%d", rag_used, len(prompt_ids))
        return prompt_text, prompt_ids, info


# ── quick smoke-test ----------------------------------------------------------
if __name__ == "__main__":
    import json, pathlib
    from tokenizer import NL2SQLTokenizer

    root = pathlib.Path(__file__).parent
    try:
        schema = json.loads((root / "sample_schema.json").read_text())
        tok = NL2SQLTokenizer(root / "nl2sql_tok.model", formatter.SPECIAL_TOKENS)
        orch = SchemaRAGOrchestrator(tok, schema)
        ptxt, pids, meta = orch.build_prompt("Which countries saw a >10% population increase in 2020?")
        print(meta, "\n", ptxt[:140], "…")
    except FileNotFoundError:
        print("Smoke-test skipped (sample files missing).")