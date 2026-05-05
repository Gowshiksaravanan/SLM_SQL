# Build Stage 2 GRPO training dataset from SynSQL

import argparse
import json
import random
import sqlite3
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

SYNSQL_JSONL  = ROOT / "pre-processing" / "stage2" / "data" / "synsql_stage2.jsonl"
EMPTY_DB_FILE = ROOT / "pre-processing" / "database_analysis" / "data" / "empty_databases.json"
SYNSQL_DB_DIR = ROOT / "data" / "SynSQL-2.5M" / "databases" / "databases"
SPIDER_DB_DIR = ROOT / "data" / "spider_official" / "spider_data" / "database"
KAGGLE_DB_DIR = ROOT / "version2" / "data" / "databases"
OUTPUT_FILE   = ROOT / "version3" / "data" / "stage2_grpo.jsonl"

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a natural language question, "
    "think step-by-step inside <think>...</think> tags to identify the relevant tables "
    "and columns, reason through the query logic, then write the final SQLite SQL query."
)


def get_db_path(db_id: str) -> Path | None:
    for base in (SYNSQL_DB_DIR, SPIDER_DB_DIR, KAGGLE_DB_DIR):
        p = base / db_id / f"{db_id}.sqlite"
        if p.exists():
            return p
    return None


def get_ddl(db_path: Path) -> str:
    try:
        con = sqlite3.connect(str(db_path))
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
        con.close()
        return "\n\n".join(r[0] for r in rows)
    except Exception:
        return ""


def validate_gold_sql(sql: str, db_path: Path, timeout: int = 5) -> bool:
    result = [None]
    def _run():
        try:
            con = sqlite3.connect(str(db_path))
            con.execute(sql).fetchall()
            con.close()
            result[0] = True
        except Exception:
            result[0] = False
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    return result[0] is True


def build_record(row: dict, ddl: str) -> dict:
    question = row["question"]
    knowledge = row.get("external_knowledge", "").strip()
    user_content = f"**Schema:**\n{ddl}\n\n**Question:** {question}"
    if knowledge:
        user_content += f"\n\n**Additional context:** {knowledge}"

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "sql":   row["sql"],
        "db_id": row["db_id"],
        "ddl":   ddl,
    }


def main(target: int, seed: int) -> None:
    random.seed(seed)

    # Load empty DB exclusion set
    print(f"Loading empty DB list from {EMPTY_DB_FILE} ...")
    with open(EMPTY_DB_FILE) as f:
        empty_data = json.load(f)
    db_list = empty_data if isinstance(empty_data, list) else empty_data.get("empty_databases", [])
    empty_db_ids: set[str] = {e.get("db_id") or e.get("db_name") for e in db_list}
    print(f"  {len(empty_db_ids):,} empty DBs to exclude")

    _ddl_cache: dict[str, str] = {}
    _db_path_cache: dict[str, Path | None] = {}
    records: list[dict] = []
    skipped_complexity = 0
    skipped_db = 0
    skipped_invalid_sql = 0

    print(f"Streaming {SYNSQL_JSONL} (Highly Complex only) ...")
    with open(SYNSQL_JSONL) as f:
        for i, line in enumerate(f):
            if (i + 1) % 100_000 == 0:
                print(f"  {i+1:,} lines  |  collected={len(records):,}  "
                      f"skip_complexity={skipped_complexity:,}  "
                      f"skip_db={skipped_db:,}  skip_sql={skipped_invalid_sql:,}")

            row = json.loads(line)

            # Hard filter: Highly Complex only
            if row.get("sql_complexity") != "Highly Complex":
                skipped_complexity += 1
                continue

            db_id = row["db_id"]

            if db_id in empty_db_ids:
                skipped_db += 1
                continue

            if db_id not in _db_path_cache:
                _db_path_cache[db_id] = get_db_path(db_id)

            db_path = _db_path_cache[db_id]
            if db_path is None:
                skipped_db += 1
                continue

            if db_id not in _ddl_cache:
                _ddl_cache[db_id] = get_ddl(db_path)

            ddl = _ddl_cache[db_id]
            if not ddl:
                skipped_db += 1
                continue

            # Validate gold SQL executes cleanly — skip records with broken gold
            if not validate_gold_sql(row["sql"], db_path):
                skipped_invalid_sql += 1
                continue

            records.append(build_record(row, ddl))

    print(f"\nCollection done:")
    print(f"  Highly Complex collected : {len(records):,}")
    print(f"  Skipped (not HC)         : {skipped_complexity:,}")
    print(f"  Skipped (no DB/DDL)      : {skipped_db:,}")
    print(f"  Skipped (invalid SQL)    : {skipped_invalid_sql:,}")

    if len(records) < target:
        print(f"  WARNING: only {len(records):,} records available, target was {target:,}")

    random.shuffle(records)
    selected = records[:target]
    print(f"\nFinal dataset: {len(selected):,} records  (target={target:,})")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for record in selected:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50_000,
                        help="Target number of records (default 50000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.target, args.seed)
