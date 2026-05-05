

import re
import sqlite3
import threading
from pathlib import Path

import sqlglot
import sqlglot.expressions as exp

ROOT = Path(__file__).parent.parent.parent.parent

SYNSQL_DB_DIR = ROOT / "data" / "SynSQL-2.5M" / "databases"
SPIDER_DB_DIR = ROOT / "data" / "spider_official" / "spider_data" / "database"

DB_SEARCH_DIRS = [SYNSQL_DB_DIR, SPIDER_DB_DIR]


def get_db_path(db_id: str) -> Path | None:
    for base in DB_SEARCH_DIRS:
        p = base / db_id / f"{db_id}.sqlite"
        if p.exists():
            return p
    return None


def strip_set_op_parens(sql: str) -> str:
    result = []
    i, n = 0, len(sql)
    while i < n:
        if sql[i] == '(':
            depth, j = 0, i
            while j < n:
                if sql[j] == '(':
                    depth += 1
                elif sql[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = sql[i + 1 : j].strip()
            after = sql[j + 1 :].lstrip()
            set_op_next = bool(re.match(r'^(INTERSECT|UNION(?:\s+ALL)?|EXCEPT)\b', after, re.IGNORECASE))
            is_end = not after.strip()
            if inner.upper().startswith('SELECT') and (set_op_next or is_end):
                result.append(inner)
            else:
                result.append(sql[i : j + 1])
            i = j + 1
        else:
            result.append(sql[i])
            i += 1
    return ''.join(result).strip()


def normalize_sql_for_execution(sql: str) -> str:# Pre-step: strip outer parens from set-op operands.SELECT A) INTERSECT (SELECT B) → SELECT A INTERSECT SELECT B.SQLite rejects parenthesized set-op operands; strip them before parsing.
    if sql.strip().startswith('(') and re.search(r'\b(?:INTERSECT|UNION|EXCEPT)\b', sql, re.IGNORECASE):
        sql = strip_set_op_parens(sql)

    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return sql

    outer_select = tree.find(exp.Select)
    if outer_select is None:
        return sql

    alias_map: dict[str, exp.Expression] = {}# Build alias_name → underlying expression map
    for item in outer_select.expressions:
        if isinstance(item, exp.Alias):
            alias_map[item.alias.lower()] = item.this

    if not alias_map:
        return sql

    for clause_type in (exp.Having, exp.Order, exp.Group, exp.Where):# Replace alias column refs in non-SELECT clauses
        clause = tree.find(clause_type)
        if clause is None:
            continue
        for col in list(clause.find_all(exp.Column)):
            if col.name.lower() in alias_map:
                col.replace(alias_map[col.name.lower()].copy())

    for item in list(outer_select.expressions):# Strip AS from outer SELECT items
        if isinstance(item, exp.Alias):
            item.replace(item.this)

    try:
        return tree.sql(dialect="sqlite")
    except Exception:
        return sql

# The training data for SynSQL uses capital letters in string literals ('Dog', 'Cat')
# while Spider stores lowercase ('dog', 'cat'). Without COLLATE NOCASE, properly formed
# SQL queries will yield 0 rows on Spider -> this is a false negative in the reward.
# This applies to gold and pred equally to make sure that the comparison is fair.
# CASE-insensitive only to string equality operators -> LIKE is already CASE-insensitive.
NOCASE_RE = re.compile(
    r"""((?:=|!=|<>)\s*(?:'(?:[^']|'')*'|"(?:[^"]|"")*"))(?!\s+COLLATE)""",
    re.IGNORECASE,
)

def add_nocase_collation(sql: str) -> str:
    return NOCASE_RE.sub(r"\1 COLLATE NOCASE", sql)

def run_sql(sql: str, db_path: Path, timeout: int = 10) -> tuple[list[dict], str | None]:
    sql = add_nocase_collation(sql)
    result: list = [None, None]

    def target():
        try:
            con = sqlite3.connect(str(db_path), timeout=timeout)
            con.row_factory = sqlite3.Row
            cur = con.execute(sql)
            result[0] = [dict(r) for r in cur.fetchall()]
            con.close()
        except Exception as e:
            result[1] = str(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout + 1)
    if t.is_alive():
        return [], "TIMEOUT"
    return result[0] or [], result[1]

def classify_error(err: str) -> str:
# Each category maps to a different penalty tier in execution.py.
# syntax and missing_table/column get the harshest negative scores
# because they indicate the model hallucinated schema entities, not just wrong logic.
    err = err.lower()
    if "syntax" in err: return "syntax"
    if "no such table" in err: return "missing_table"
    if "no such column" in err: return "missing_column"
    if "timeout" in err: return "timeout"
    return "runtime"


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SQL_RE= re.compile(r"```sql\s*(.*?)```",    re.DOTALL)


def completion_text(completion) -> str:
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    return completion


# </think> split takes priority over sql because the model is trained to put SQL directly after the closing think tag — no markdown fence needed. The sql fallback handles edge cases where the model wraps output in a code block instead.
def extract_sql(completion) -> str:
    text = completion_text(completion)
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    m = SQL_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def extract_cot(completion) -> str:
    text = completion_text(completion)
    m = THINK_RE.search(text)
    return m.group(1).strip() if m else ""

def normalize_value(v):
    if v is None:
        return None
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return str(v).lower().strip()


def extract_tables_cols(sql: str) -> tuple[set[str], set[str]]:
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
        tables = {t.name.lower() for t in parsed.find_all(exp.Table)  if t.name}
        cols   = {c.name.lower() for c in parsed.find_all(exp.Column) if c.name}
        return tables, cols
    except Exception:
        return set(), set()
# Regex first because it's faster and the DDL strings come from our own SQLite extraction (well-formed). sqlglot fallback handles edge cases like quoted identifiers or non-standard formatting that the regex misses.
CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?\s*\(([^;]*)\)",
    re.IGNORECASE | re.DOTALL,
)
COL_NAME_RE = re.compile(r"^\s*[`\"\[]?(\w+)[`\"\]]?\s+\w", re.IGNORECASE)


def parse_schema_entities(ddl: str) -> tuple[set[str], set[str]]:
    tables: set[str] = set()
    cols:   set[str] = set()

    for m in CREATE_TABLE_RE.finditer(ddl):
        tables.add(m.group(1).lower())
        for line in m.group(2).splitlines():
            line = line.strip().rstrip(",")
            if re.match(r"(PRIMARY|FOREIGN|UNIQUE|CHECK|INDEX|KEY)\s", line, re.I):
                continue
            cm = COL_NAME_RE.match(line)
            if cm:
                cols.add(cm.group(1).lower())

    if not tables:  # sqlglot fallback
        try:
            for stmt in sqlglot.parse(ddl, dialect="sqlite"):
                if isinstance(stmt, exp.Create):
                    tbl = stmt.find(exp.Table)
                    if tbl:
                        tables.add(tbl.name.lower())
                    for col_def in stmt.find_all(exp.ColumnDef):
                        cols.add(col_def.name.lower())
        except Exception:
            pass

    return tables, cols
