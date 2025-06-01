"""
selector.py – Table retrieval with *dynamic* similarity threshold.

Key points
----------
1. Dynamic threshold = max(base, 0.5 * max_similarity)
   • Always keeps at least the single most relevant table.
   • Scales up automatically for generic / short questions.

2. If dynamic threshold still yields an empty set, we fall back to
   **top-1** table (guaranteeing ≥1).

Other logic (FK expansion, batching, logging) unchanged.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from schema_pipeline.graph import COLUMN_PREFIX

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ────────────────────────────────────────────────────────────────────────────
def _table_context(table: Dict) -> str:
    return f"{table['name']} " + " ".join(f"{c['name']}:{c['type']}" for c in table["columns"])


def build_table_embeddings(schema: Dict, model: SentenceTransformer) -> Dict[str, np.ndarray]:
    names, texts = [], []
    for t in schema["tables"]:
        names.append(t["name"])
        texts.append(_table_context(t))
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return dict(zip(names, vecs))


def _fk_neighbours(G, table_name: str) -> List[str]:
    neigh = set()
    for u, v, data in G.edges(data=True):
        if data.get("rel") != "FK":
            continue
        for node in (u, v):
            if node.startswith(COLUMN_PREFIX) and G.nodes[node]["table"] == table_name:
                neigh.add(G.nodes[(v if node == u else u)]["table"])
    return list(neigh)


# ────────────────────────────────────────────────────────────────────────────
def select_tables(
    question: str,
    G,
    table_embeddings: Dict[str, np.ndarray],
    model: SentenceTransformer,
    base_threshold: float = 0.20,
) -> List[str]:
    """
    Dynamic-threshold selection + FK expansion.

    Returns list of unique table names (similarity-desc sorted).
    """
    q_vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    names = list(table_embeddings)
    mat = np.vstack([table_embeddings[n] for n in names])
    sims = (mat @ q_vec).astype(float)  # cosine because vectors L2-normalised

    # 1) Dynamic threshold — keeps ≥1 table by design
    dyn_thresh = max(base_threshold, 0.5 * sims.max())
    primary = [n for n, s in zip(names, sims) if s >= dyn_thresh]

    if not primary:  # extreme edge-case
        primary = [names[int(sims.argmax())]]

    logger.debug("Selector | dyn_thresh=%.3f | primary=%s", dyn_thresh, primary)

    # 2) FK expansion --------------------------------------------------------
    expanded = set(primary)
    for t in primary:
        expanded.update(_fk_neighbours(G, t))

    result = sorted(expanded, key=lambda n: (-table_embeddings[n] @ q_vec, n))
    logger.info("Selector | final tables=%s", result)
    return result