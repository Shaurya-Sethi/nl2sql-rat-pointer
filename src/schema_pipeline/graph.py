"""
graph.py
========
Build a directed NetworkX graph from an extracted database *schema* and
capture **composite primary keys**.

# All imports in this file should be absolute, relative to src/ root if needed.
# Do not use sys.path hacks or relative imports.

Node IDs
--------
table  →  "table:<table_name>"
column →  "column:<table_name>.<column_name>"

Node attributes
---------------
Table:
    kind          = "table"
    table         = <table_name>
    pk_columns    = [col1, col2, ...]          # NEW
    composite_pk  = bool                       # NEW  (True ⇢ len(pk_columns) > 1)

Column:
    kind       = "column"
    table      = <table_name>
    column     = <column_name>
    data_type  = <SQL type>
    pk         = bool

Edges
-----
table   ─▶ column                 rel="HAS_COLUMN"
column  ─▶ column  (FK mapping)   rel="FK"
"""

from __future__ import annotations

import logging
from typing import Dict, List

import networkx as nx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TABLE_PREFIX  = "table:"
COLUMN_PREFIX = "column:"


def _table_id(name: str) -> str:
    return f"{TABLE_PREFIX}{name}"


def _column_id(table: str, column: str) -> str:
    return f"{COLUMN_PREFIX}{table}.{column}"


# ────────────────────────────────────────────────────────────────────────────
def build_schema_graph(schema: Dict) -> nx.DiGraph:
    """
    Parameters
    ----------
    schema : Dict
        Dict produced by *extractor.extract_schema()*.

    Returns
    -------
    nx.DiGraph
        Graph with rich node metadata and FK edges.  Supports composite PKs.
    """
    G: nx.DiGraph = nx.DiGraph()

    # ── Pass 1: Tables & columns ────────────────────────────────────────────
    for table in schema.get("tables", []):
        tname: str = table["name"]
        t_node     = _table_id(tname)

        # Identify ALL PK columns for this table
        pk_cols: List[str] = [col["name"] for col in table["columns"] if col.get("pk")]
        G.add_node(
            t_node,
            kind="table",
            table=tname,
            pk_columns=pk_cols,
            composite_pk=len(pk_cols) > 1,      # <-- NEW flag
        )

        for col in table["columns"]:
            c_node = _column_id(tname, col["name"])
            G.add_node(
                c_node,
                kind="column",
                table=tname,
                column=col["name"],
                data_type=col["type"],
                pk=bool(col.get("pk", False)),
            )
            G.add_edge(t_node, c_node, rel="HAS_COLUMN")

    # ── Pass 2: Foreign-key edges ───────────────────────────────────────────
    for fk in schema.get("foreign_keys", []):
        src = _column_id(fk["from_table"], fk["from_column"])
        dst = _column_id(fk["to_table"],   fk["to_column"])
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst, rel="FK")

    logger.debug(
        "Graph built | %d nodes | %d edges | composite-PK tables: %d",
        G.number_of_nodes(),
        G.number_of_edges(),
        sum(1 for _, d in G.nodes(data=True) if d.get("composite_pk")),
    )
    return G


# ── quick self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal synthetic example with a composite PK
    demo_schema = {
        "tables": [
            {
                "name": "enrollments",
                "columns": [
                    {"name": "student_id", "type": "INTEGER", "pk": True},
                    {"name": "course_id",  "type": "INTEGER", "pk": True},
                    {"name": "enrolled_on","type": "DATE",    "pk": False},
                ],
            }
        ],
        "foreign_keys": [],
    }

    G = build_schema_graph(demo_schema)
    tbl_meta = G.nodes["table:enrollments"]
    assert tbl_meta["composite_pk"] is True and tbl_meta["pk_columns"] == ["student_id", "course_id"]
    print("composite-PK handled correctly →", tbl_meta["pk_columns"])