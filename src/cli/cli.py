import argparse
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import traceback

from getpass import getpass
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table

# --- Transqlate internals ----------------------------------------------------
from extractor import get_schema_extractor
from formatter import format_schema, SPECIAL_TOKENS as FMT_TOK
from tokenizer import NL2SQLTokenizer
from orchestrator import SchemaRAGOrchestrator
from selector import build_table_embeddings
from inference import NL2SQLInference

# ───────── Optional fuzzy/embeddings libs ─────────
try:
    from rapidfuzz import process as fuzz_process  # noqa: E401
except ImportError:
    fuzz_process = None

console = Console()

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def print_exception(exc: Exception):
    console.print(Panel.fit(f"[bold red]Error:[/bold red] {exc}", style="red"))
    if console.is_verbose:
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Interactive credential gathering
# -----------------------------------------------------------------------------

def collect_db_params(db_type: str) -> Tuple[str, dict]:
    params = {}
    if db_type == "sqlite":
        params["db_path"] = Prompt.ask("SQLite file path", default="example.db")
    else:
        params["host"] = Prompt.ask("Host", default="localhost")
        default_port = {
            "postgres": "5432",
            "postgresql": "5432",
            "mysql": "3306",
            "mssql": "1433",
            "oracle": "1521",
        }.get(db_type, "5432")
        params["port"] = int(Prompt.ask("Port", default=default_port))
        if db_type in {"postgres", "postgresql"}:
            params["dbname"] = Prompt.ask("Database name")
        else:
            params["database"] = Prompt.ask("Database name")
        params["user"] = Prompt.ask("Username")
        params["password"] = getpass("Password: ")
        if db_type == "oracle":
            params["service_name"] = params.pop("database")
    return db_type, params


def choose_db_interactively() -> Tuple[str, dict]:
    db_type = Prompt.ask(
        "Choose DB type",
        choices=["sqlite", "postgres", "mysql", "mssql", "oracle"],
        default="sqlite",
    )
    return collect_db_params(db_type)


# -----------------------------------------------------------------------------
# Session state dataclass – keeps connections & objects around
# -----------------------------------------------------------------------------

class Session:
    def __init__(
        self,
        db_type: str,
        extractor,
        schema_dict: dict,
        tokenizer: NL2SQLTokenizer,
        orchestrator: SchemaRAGOrchestrator,
        inference: NL2SQLInference,
        table_embs=None,
    ):
        self.db_type = db_type
        self.extractor = extractor
        self.schema_dict = schema_dict
        self.tokenizer = tokenizer
        self.orchestrator = orchestrator
        self.inference = inference
        self.history: List[Tuple[str, str]] = []  # (question, sql)
        self.table_embs = table_embs

    def execute_sql(self, sql: str):
        try:
            cur = self.extractor.conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            self._pretty_table(rows, cols)
        except Exception as e:
            print_exception(e)

    def _pretty_table(self, rows, cols):
        if not rows:
            console.print("[yellow]No rows returned.[/yellow]")
            return
        table = Table(show_header=True, header_style="bold magenta")
        for c in cols:
            table.add_column(str(c))
        for r in rows:
            table.add_row(*[str(cell) for cell in r])
        console.print(table)

    def suggest_schema_terms(self, token: str, top_k: int = 5) -> List[str]:
        names = [t["name"] for t in self.schema_dict["tables"]]
        cols = [f"{t['name']}.{c['name']}" for t in self.schema_dict["tables"] for c in t["columns"]]
        candidates = names + cols
        suggestions = []
        if fuzz_process is not None:
            suggestions = [s for s, _ in fuzz_process.extract(token, candidates, limit=top_k)]
        return suggestions


# -----------------------------------------------------------------------------
# Build session
# -----------------------------------------------------------------------------

def build_session(args) -> Optional[Session]:
    if args.db_type:
        db_type = args.db_type.lower()
        params = {k: v for k, v in {
            "db_path": args.db_path,
            "host": args.host,
            "port": args.port,
            "dbname": args.database,
            "database": args.database,
            "user": args.user,
            "password": args.password,
        }.items() if v is not None}
    else:
        db_type, params = choose_db_interactively()
    try:
        extractor = get_schema_extractor(db_type, **params)
        schema_dict = extractor.extract_schema()
        console.print(f"[green]\u2714 Connected to {db_type} database.[/green]")
    except Exception as e:
        print_exception(e)
        return None
    tok_path = args.sp_model or "models/nl2sql_tok.model"
    tokenizer = NL2SQLTokenizer(tok_path, FMT_TOK)
    orchestrator = SchemaRAGOrchestrator(tokenizer, schema_dict)
    model_ckpt = args.model or "models/ckpt.pt"
    config_path = args.config or "src/config.yaml"
    inference = NL2SQLInference(model_ckpt, config_path, tokenizer_path=tok_path)
    table_embs = build_table_embeddings(schema_dict, orchestrator._embed)
    return Session(db_type, extractor, schema_dict, tokenizer, orchestrator, inference, table_embs)


# -----------------------------------------------------------------------------
# REPL with streaming output
# -----------------------------------------------------------------------------

def repl(session: Session, run_sql: bool):
    console.print(Panel("[bold cyan]Transqlate[/bold cyan] – Natural Language → SQL", title="Welcome", expand=False))
    console.print("Type your natural language query or :help for commands.\n")
    while True:
        try:
            line = Prompt.ask("[bold green]Transqlate ›[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]Exiting…[/red]")
            break
        if not line.strip():
            continue
        if line.startswith(":"):
            cmd, *rest = line[1:].strip().split()
            if cmd in {"exit", "quit", "q"}:
                break
            elif cmd == "help":
                console.print(
                    ":help – show this help\n"
                    ":history – show past queries\n"
                    ":show schema – print formatted schema\n"
                    ":run – re-run last SQL against DB\n"
                    ":examples – sample NL prompts\n"
                    ":clear – clear screen\n"
                    ":exit – quit",
                    style="cyan",
                )
            elif cmd == "history":
                for i, (q, s) in enumerate(session.history[-10:], 1):
                    console.print(f"[yellow]{i}.[/yellow] {q} → [cyan]{s}[/cyan]")
            elif cmd == "show" and rest and rest[0] == "schema":
                console.print(format_schema(session.schema_dict))
            elif cmd == "run":
                if not session.history:
                    console.print("[yellow]No previous query to run.[/yellow]")
                else:
                    session.execute_sql(session.history[-1][1])
            elif cmd == "examples":
                console.print(
                    "- Show me total sales by month in 2023\n"
                    "- List top 5 customers by revenue\n"
                    "- Average delivery time per city",
                    style="dim",
                )
            elif cmd == "clear":
                console.clear()
            else:
                console.print(f"[red]Unknown command[/red]: {cmd}")
            continue
        with console.status("Processing…", spinner="dots"):
            try:
                _, prompt_ids, _ = session.orchestrator.build_prompt(line)
                sch_id = session.tokenizer.special_tokens["<SCHEMA>"]
                end_id = session.tokenizer.special_tokens["</SCHEMA>"]
                start = prompt_ids.index(sch_id)
                end = prompt_ids.index(end_id)
                schema_tokens = prompt_ids[start:end+1]
                stream = session.inference.infer_stream(line, schema_tokens)
            except Exception as e:
                msg = str(e)
                m = re.search(r"(?:no such column|Unknown column|column\s+)(?:\s|')(\w+)", msg, re.I)
                if m:
                    token = m.group(1)
                    sugg = session.suggest_schema_terms(token)
                    if sugg:
                        console.print(
                            Panel(
                                f"Could not find [bold]{token}[/bold]. Did you mean: "
                                + ", ".join(f"[cyan]{s}[/cyan]" for s in sugg)
                                + "?",
                                style="yellow",
                            )
                        )
                        new_tok = Prompt.ask("Correction", default=sugg[0])
                        if new_tok.strip() and new_tok != token:
                            line = line.replace(token, new_tok)
                            session.history.append((f"(ambig {token}->{new_tok}) {line}", ""))
                            continue
                print_exception(e)
                continue
        # Streaming CoT & SQL
        console.print("\n[dim]Chain of Thought:[/dim]")
        in_cot = False
        in_sql = False
        cot_text = ""
        sql_text = ""
        for token in stream:
            if token == "<COT>":
                in_cot = True
                continue
            if token == "</COT>":
                in_cot = False
                continue
            if token == "<SQL>":
                console.print()  
                console.print("\n[bold cyan]SQL:[/bold cyan]", end=" ")
                in_sql = True
                continue
            if token == "</SQL>":
                break
            if in_cot:
                console.print(token, end="", style="dim")
                cot_text += token
            elif in_sql:
                console.print(token, end="", style="bold cyan")
                sql_text += token
        console.print()  
        session.history.append((line, sql_text))
        if run_sql:
            session.execute_sql(sql_text)


# -----------------------------------------------------------------------------
# One-shot mode with streaming
# -----------------------------------------------------------------------------

def oneshot(session: Session, question: str, execute: bool):
    line = question
    _, prompt_ids, _ = session.orchestrator.build_prompt(line)
    sch_id = session.tokenizer.special_tokens["<SCHEMA>"]
    end_id = session.tokenizer.special_tokens["</SCHEMA>"]
    start = prompt_ids.index(sch_id)
    end = prompt_ids.index(end_id)
    schema_tokens = prompt_ids[start:end+1]
    console.print(Panel(f"[bold green]Query:[/bold green] {line}"))
    with console.status("Processing…", spinner="dots"):
        stream = session.inference.infer_stream(line, schema_tokens)
    console.print("\n[dim]Chain of Thought:[/dim]")
    in_cot = False
    in_sql = False
    cot_text = ""
    sql_text = ""
    for token in stream:
        if token == "<COT_START>":
            in_cot = True
            continue
        if token == "<COT_END>":
            in_cot = False
            continue
        if token == "<SQL_START>":
            console.print()  
            console.print("\n[bold cyan]SQL:[/bold cyan]", end=" ")
            in_sql = True
            continue
        if token == "</SQL>":
            break
        if in_cot:
            console.print(token, end="", style="dim")
            cot_text += token
        elif in_sql:
            console.print(token, end="", style="bold cyan")
            sql_text += token
    console.print()  
    if execute:
        session.execute_sql(sql_text)


def main():
    parser = argparse.ArgumentParser("transqlate – Natural Language to SQL CLI")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive REPL mode")
    parser.add_argument("--question", "-q", help="One-shot natural language question")
    parser.add_argument("--execute", action="store_true", help="Execute generated SQL and show results")
    parser.add_argument("--db-type", choices=["sqlite","postgres","mysql","mssql","oracle"], help="DB type")
    parser.add_argument("--db-path", help="SQLite path")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--database")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--model", help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", help="Path to config YAML (default: src/config.yaml)")
    parser.add_argument("--sp-model", help="Path to SentencePiece model")
    args = parser.parse_args()
    if not args.interactive and not args.question:
        parser.error("Provide --interactive or --question/-q for one-shot mode.")
    session = build_session(args)
    if not session:
        sys.exit(1)
    if args.interactive:
        repl(session, run_sql=args.execute)
    else:
        oneshot(session, question=args.question, execute=args.execute)

if __name__ == "__main__":
    main()