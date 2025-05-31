"""
universal_schema_extractor.py  (robust v2)
------------------------------------------
• Unified, multi-DB schema extractor
• Adds error-handling, cursor auto-close, identifier normalisation, and
  safer SQL execution.

Dependencies (install what you need):
    pip install psycopg2-binary mysql-connector-python pyodbc cx_Oracle
"""
from __future__ import annotations
import sqlite3
import logging
from typing import Dict, List, Any

# ───────── Optional drivers ─────────
try:
    import psycopg2, psycopg2.sql as psql
except ImportError:
    psycopg2 = psql = None

try:
    import mysql.connector
except ImportError:
    mysql = None

# MS-SQL
try:
    import pyodbc
    _mssql_driver = "pyodbc"
except ImportError:           # fallback
    try:
        import pymssql
        _mssql_driver = "pymssql"
    except ImportError:
        _mssql_driver = None

# Oracle
try:
    import cx_Oracle
except ImportError:
    cx_Oracle = None
# ────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("schema_extractor")

# ═════════ Custom error ═════════════
class SchemaExtractionError(RuntimeError):
    """Raised when a query/connection problem prevents schema extraction."""


# ═════════ Helper utils ═════════════
def _bytes2str(value: Any) -> Any:
    """Convert MySQL bytes columns → str (utf-8) so JSON is safe."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode('utf-8', errors='replace')
    return value


def _normalize_identifier(dbms: str, identifier: str) -> str:
    """Return identifier in the default case rules of each DBMS."""
    if dbms == "postgresql":
        return identifier.lower()
    if dbms == "oracle":
        return identifier.upper()
    return identifier


# ═════════ Base extractor ═══════════
class BaseSchemaExtractor:
    dbms: str  # "sqlite" / "postgresql" / "mysql" / "mssql" / "oracle"

    # ---- mandatory API --------------
    def get_tables(self) -> List[str]: ...
    def get_columns(self, table: str) -> List[Dict]: ...
    def get_foreign_keys(self, table: str) -> List[Dict]: ...

    # ---- common helpers -------------
    def _safe_exec(self, query: str, params: tuple | dict | None = None,
                   fetch: str = "all") -> list:
        """
        Run a query safely, close cursor manually, convert bytes→str,
        re-raise SchemaExtractionError on failure.
        """
        cur = None
        try:
            cur = self.conn.cursor()
            cur.execute(query, params or ())
            rows = cur.fetchall() if fetch == "all" else cur.fetchone()
            return [[_bytes2str(v) for v in row] for row in rows]
        except Exception as exc:
            logger.error("%s query failed: %s", self.dbms.upper(), exc)
            raise SchemaExtractionError(str(exc)) from exc
        finally:
            if cur is not None:
                try:
                    cur.close()
                except Exception:
                    pass

    # ---- convenience ----------------
    def extract_schema(self) -> Dict:
        schema = {"tables": [], "foreign_keys": []}
        for tbl in self.get_tables():
            schema["tables"].append({"name": tbl,
                                     "columns": self.get_columns(tbl)})
            schema["foreign_keys"].extend(self.get_foreign_keys(tbl))
        return schema

    def close(self):
        try:
            self.conn.close()
        except Exception:     # pragma: no cover
            pass

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


# ═════════ 1. SQLite ════════════════
class SQLiteSchemaExtractor(BaseSchemaExtractor):
    dbms = "sqlite"

    def __init__(self, db_path: str):
        logger.info("SQLite | connecting to %s", db_path)
        self.conn = sqlite3.connect(db_path)

    def get_tables(self):
        rows = self._safe_exec(
            "SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in rows]

    def get_columns(self, table):
        rows = self._safe_exec(f"PRAGMA table_info({table})")
        return [{"name": r[1], "type": r[2], "pk": bool(r[5])} for r in rows]

    def get_foreign_keys(self, table):
        rows = self._safe_exec(f"PRAGMA foreign_key_list({table})")
        return [{"from_table": table, "from_column": r[3],
                 "to_table": r[2], "to_column": r[4]} for r in rows]


# ═════════ 2. PostgreSQL ════════════
class PostgresSchemaExtractor(BaseSchemaExtractor):
    dbms = "postgresql"

    def __init__(self, host, user, password, dbname, port=5432, schema="public"):
        if psycopg2 is None:
            raise ImportError("psycopg2 is required for PostgreSQL support.")
        logger.info("PostgreSQL | %s:%s/%s (schema=%s)", host, port, dbname, schema)
        self.schema = schema
        self.conn = psycopg2.connect(host=host, user=user, password=password,
                                     dbname=dbname, port=port)

    def get_tables(self):
        q = psql.SQL("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_type='BASE TABLE';
        """)
        rows = self._safe_exec(q, (self.schema,))
        return [r[0] for r in rows]

    def get_columns(self, table):
        table_norm = _normalize_identifier(self.dbms, table)
        q = psql.SQL(\"\"\"
            SELECT c.column_name, c.data_type,
                   EXISTS (
                     SELECT 1 FROM information_schema.table_constraints tc
                     JOIN information_schema.key_column_usage kcu
                       ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
                     WHERE tc.table_name = %s
                       AND tc.table_schema = %s  -- Added schema filter for tc
                       AND tc.constraint_type = 'PRIMARY KEY'
                       AND kcu.column_name = c.column_name
                   ) AS is_pk
            FROM information_schema.columns c
            WHERE c.table_name = %s AND c.table_schema = %s; -- Added schema filter for c
        \"\"\")
        rows = self._safe_exec(q, (table_norm, self.schema, table_norm, self.schema))
        return [{"name": r[0], "type": r[1], "pk": bool(r[2])} for r in rows]

    def get_foreign_keys(self, table):
        table_norm = _normalize_identifier(self.dbms, table)
        q = psql.SQL(\"\"\"
            SELECT kcu.table_name, kcu.column_name,
                   ccu.table_name, ccu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type='FOREIGN KEY'
              AND tc.table_name=%s
              AND tc.table_schema = %s; -- Added schema filter for tc
        \"\"\")
        rows = self._safe_exec(q, (table_norm, self.schema))
        return [{"from_table": r[0], "from_column": r[1],
                 "to_table": r[2], "to_column": r[3]} for r in rows]


# ═════════ 3. MySQL / MariaDB ═══════
class MySQLSchemaExtractor(BaseSchemaExtractor):
    dbms = "mysql"

    def __init__(self, host, user, password, database, port=3306):
        if mysql is None:
            raise ImportError("mysql-connector-python required for MySQL.")
        logger.info("MySQL | %s:%s/%s", host, port, database)
        self.conn = mysql.connector.connect(
            host=host, user=user, password=password,
            database=database, port=port)

    def get_tables(self):
        rows = self._safe_exec("SHOW TABLES")
        return [r[0] for r in rows]

    def get_columns(self, table):
        rows = self._safe_exec(f"DESCRIBE `{table}`")
        return [{"name": r[0], "type": _bytes2str(r[1]),
                 "pk": (r[3] == "PRI")} for r in rows]

    def get_foreign_keys(self, table):
        q = """
            SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME   = %s
              AND REFERENCED_TABLE_NAME IS NOT NULL;
        """
        rows = self._safe_exec(q, (table,))
        return [{"from_table": table, "from_column": r[0],
                 "to_table": r[1], "to_column": r[2]} for r in rows]


# ═════════ 4. MS SQL Server ═════════
class MSSQLSchemaExtractor(BaseSchemaExtractor):
    dbms = "mssql"

    def __init__(self, host, user, password, database,
                 port=1433, driver=None):
        if _mssql_driver is None:
            raise ImportError("pyodbc or pymssql required for SQL Server.")
        logger.info("MS-SQL | %s:%s/%s via %s", host, port, database, _mssql_driver)

        if _mssql_driver == "pyodbc":
            driver = driver or "{ODBC Driver 17 for SQL Server}"
            conn_str = (f"DRIVER={driver};SERVER={host},{port};"
                        f"DATABASE={database};UID={user};PWD={password};"
                        "TrustServerCertificate=yes")
            self.conn = pyodbc.connect(conn_str)
        else:  # pymssql
            self.conn = pymssql.connect(server=host, user=user,
                                        password=password, database=database,
                                        port=port)

    # helpers
    def _rows(self, query, params=()):
        return self._safe_exec(query, params)

    def get_tables(self):
        rows = self._rows("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE='BASE TABLE';
        """)
        return [r[0] for r in rows]

    def get_columns(self, table):
        rows = self._rows("""
            SELECT c.COLUMN_NAME, c.DATA_TYPE,
                   CASE WHEN k.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN (
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
                JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                  ON k.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
                WHERE tc.TABLE_NAME=%s AND tc.CONSTRAINT_TYPE='PRIMARY KEY'
            ) k ON k.COLUMN_NAME = c.COLUMN_NAME
            WHERE c.TABLE_NAME=%s
        """, (table, table))
        return [{"name": r[0], "type": r[1], "pk": bool(r[2])} for r in rows]

    def get_foreign_keys(self, table):
        rows = self._rows("""
            SELECT fk_col.COLUMN_NAME, pk_tab.TABLE_NAME, pk_col.COLUMN_NAME
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS fk_tab
                 ON rc.CONSTRAINT_NAME = fk_tab.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS pk_tab
                 ON rc.UNIQUE_CONSTRAINT_NAME = pk_tab.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE fk_col
                 ON rc.CONSTRAINT_NAME = fk_col.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE pk_col
                 ON pk_tab.CONSTRAINT_NAME = pk_col.CONSTRAINT_NAME
            WHERE fk_tab.TABLE_NAME=%s;
        """, (table,))
        return [{"from_table": table, "from_column": r[0],
                 "to_table": r[1], "to_column": r[2]} for r in rows]


# ═════════ 5. Oracle ════════════════
class OracleSchemaExtractor(BaseSchemaExtractor):
    dbms = "oracle"

    def __init__(self, host, user, password, service_name, port=1521):
        if cx_Oracle is None:
            raise ImportError("cx_Oracle required for Oracle support.")
        dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
        logger.info("Oracle | %s:%s/%s", host, port, service_name)
        self.conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)

    def get_tables(self):
        rows = self._safe_exec("SELECT table_name FROM user_tables")
        return [r[0] for r in rows]

    def get_columns(self, table):
        tbl = _normalize_identifier(self.dbms, table)
        q = """
            SELECT column_name, data_type,
                   CASE WHEN column_name IN (
                     SELECT acc.column_name
                     FROM user_constraints ac
                     JOIN user_cons_columns acc
                       ON ac.constraint_name = acc.constraint_name
                     WHERE ac.constraint_type='P' AND ac.table_name=:tbl
                   ) THEN 1 ELSE 0 END AS is_pk
            FROM user_tab_columns
            WHERE table_name=:tbl
        """
        rows = self._safe_exec(q, {"tbl": tbl})
        return [{"name": r[0], "type": r[1], "pk": bool(r[2])} for r in rows]

    def get_foreign_keys(self, table):
        tbl = _normalize_identifier(self.dbms, table)
        q = """
            SELECT a.column_name,
                   c_pk.table_name,
                   b.column_name
            FROM user_constraints c
            JOIN user_cons_columns a ON c.constraint_name = a.constraint_name
            JOIN user_cons_columns b ON c.r_constraint_name = b.constraint_name
            JOIN user_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
            WHERE c.constraint_type='R' AND c.table_name=:tbl
        """
        rows = self._safe_exec(q, {"tbl": tbl})
        return [{"from_table": table, "from_column": r[0],
                 "to_table": r[1], "to_column": r[2]} for r in rows]


# ═════════ 6. Factory ═══════════════
def get_schema_extractor(db_type: str, **kwargs) -> BaseSchemaExtractor:
    db_type = db_type.lower()
    if db_type == "sqlite":
        return SQLiteSchemaExtractor(kwargs["db_path"])
    if db_type in ("postgres", "postgresql"):
        return PostgresSchemaExtractor(
            kwargs["host"], kwargs["user"], kwargs["password"],
            kwargs["dbname"], kwargs.get("port", 5432),
            kwargs.get("schema", "public"))
    if db_type == "mysql":
        return MySQLSchemaExtractor(
            kwargs["host"], kwargs["user"], kwargs["password"],
            kwargs["database"], kwargs.get("port", 3306))
    if db_type in ("mssql", "sqlserver"):
        return MSSQLSchemaExtractor(
            kwargs["host"], kwargs["user"], kwargs["password"],
            kwargs["database"], kwargs.get("port", 1433),
            kwargs.get("driver"))
    if db_type == "oracle":
        return OracleSchemaExtractor(
            kwargs["host"], kwargs["user"], kwargs["password"],
            kwargs["service_name"], kwargs.get("port", 1521))
    raise ValueError(f"Unsupported database type: {db_type}")


# ═════════ 7. Ad-hoc manual test ════
if __name__ == "__main__":
    # Minimal quick-check — change the path/credentials to your own DB.
    db_path = r"C:\Users\shaur\NL2SQL-Transformer-Scaffold\datasets\paired_nl_sql\synsql\databases\social_media_data_management_and_analytics_706954\social_media_data_management_and_analytics_706954.sqlite"
    with get_schema_extractor("sqlite", db_path=db_path) as ex:
        import json, pprint, textwrap
        pprint.pp(print(json.dumps(ex.extract_schema(), indent=2)))