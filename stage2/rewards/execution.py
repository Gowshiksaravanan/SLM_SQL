from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config_stage2 as cfg
from rewards.utils import (
    classify_error,
    extract_sql,
    get_db_path,
    normalize_sql_for_execution,
    normalize_value,
    run_sql,
)


def f2(precision: float, recall: float) -> float:
    denom = 4 * precision + recall
    return (5 * precision * recall / denom) if denom > 0 else 0.0


def to_counter(rows: list[dict], cols: list[str]) -> Counter:
    return Counter(
        tuple(normalize_value(row.get(c)) for c in cols)
        for row in rows
    )


def f2_from_counters(gold_c: Counter, pred_c: Counter) -> float:
    intersection  = gold_c & pred_c
    inter_count   = sum(intersection.values())
    precision     = inter_count / max(sum(pred_c.values()), 1)
    recall        = inter_count / max(sum(gold_c.values()), 1)
    return f2(precision, recall)


def compare_rows(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> tuple[float, float]:
    gold_cols_raw = list(gold_rows[0].keys())
    pred_cols_raw = list(pred_rows[0].keys())

    # ── Pass 1: column-name match (case-insensitive) ──────────────────────────
    gold_cols_lower = {c.lower(): c for c in gold_cols_raw}
    pred_cols_lower = {c.lower(): c for c in pred_cols_raw}

    common_lower = set(gold_cols_lower) & set(pred_cols_lower)
    col_score    = len(common_lower) / max(len(gold_cols_raw), 1)

    if common_lower:
        gold_common = sorted(common_lower)
        pred_common = sorted(common_lower)

        gold_counter = to_counter(gold_rows, [gold_cols_lower[c] for c in gold_common])
        pred_counter = to_counter(pred_rows, [pred_cols_lower[c] for c in pred_common])
        pass1f2 = f2_from_counters(gold_counter, pred_counter)
    else:
        pass1f2 = 0.0

    # ── Pass 2: positional match (same column count) ──────────────────────────
    if len(gold_cols_raw) == len(pred_cols_raw):
        gold_counter_pos = to_counter(gold_rows, gold_cols_raw)
        pred_counter_pos = to_counter(pred_rows, pred_cols_raw)
        pass2f2 = f2_from_counters(gold_counter_pos, pred_counter_pos)
    else:
        pass2f2 = 0.0

    row_score = max(pass1f2, pass2f2)
    return row_score, col_score


def compute_single(pred_sql: str, gold_sql: str, db_path: Path) -> float:

    # Normalize aliases before execution
    gold_sql_norm = normalize_sql_for_execution(gold_sql)
    pred_sql_norm = normalize_sql_for_execution(pred_sql)

    pred_rows, pred_err = run_sql(pred_sql_norm, db_path, timeout=cfg.SQL_TIMEOUT_SECONDS)
    gold_rows, gold_err = run_sql(gold_sql_norm, db_path, timeout=cfg.SQL_TIMEOUT_SECONDS)

    # Gold failed — can't evaluate
    if gold_err:
        return 0.0

    # Pred execution failed — negative rewards
    if pred_err:
        err_type = classify_error(pred_err)
        if err_type == "syntax":        return -1.00
        if err_type == "missing_table": return -0.50
        if err_type == "missing_column":return -0.30
        return -0.20  # timeout / other runtime

    # Both empty → correct
    if len(gold_rows) == 0 and len(pred_rows) == 0:
        return 1.0

    # Mismatch on emptiness
    if len(gold_rows) > 0 and len(pred_rows) == 0:
        return -0.15  # missed the data
    if len(gold_rows) == 0 and len(pred_rows) > 0:
        return -0.10  # returned unwanted data

    # Both have rows — compare
    row_score, col_score = compare_rows(gold_rows, pred_rows)

    if col_score == 0.0 and row_score == 0.0:
        return 0.0  # no overlap at all

    # Exact match
    if row_score == 1.0 and col_score == 1.0:
        return 1.0

    # Partial match — maps to (0.0, 0.90]
    return col_score * row_score * 0.9


def execution_reward_func(
    completions: list[str],
    prompts: list[str],
    **kwargs,
) -> list[float]:
    gold_sqls = kwargs["sql"]
    db_ids    = kwargs["db_id"]

    rewards = []
    for completion, gold_sql, db_id in zip(completions, gold_sqls, db_ids):
        pred_sql = extract_sql(completion)
        db_path  = get_db_path(db_id)
        if db_path is None:
            rewards.append(0.0)
            continue
        raw = compute_single(pred_sql, gold_sql, db_path)
        rewards.append(raw * cfg.EXECUTION_WEIGHT)
    return rewards


if __name__ == "__main__":
    import sqlite3 as _sqlite3, tempfile

    db_file = Path(tempfile.mktemp(suffix=".sqlite"))
    con = _sqlite3.connect(str(db_file))
    con.executescript("""
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, amount REAL);
        INSERT INTO orders VALUES (1, 'alice', 100.0);
        INSERT INTO orders VALUES (2, 'bob',   200.0);
        INSERT INTO orders VALUES (3, 'alice',  50.0);
    """)
    con.close()

    def score(pred: str, gold: str) -> float:
        return compute_single(pred, gold, db_file)

#Formatted this case using claude
    cases = [
        # (description,                          pred_sql,                                          gold_sql,                                          lo,    hi)
        ("syntax error",                         "SELEC * FROM orders",                             "SELECT * FROM orders",                            -1.01, -0.99),
        ("missing table",                        "SELECT * FROM invoices",                          "SELECT * FROM orders",                            -0.51, -0.49),
        ("missing column",                       "SELECT ghost_col FROM orders",                    "SELECT * FROM orders",                            -0.31, -0.29),
        ("pred empty, gold has rows",            "SELECT * FROM orders WHERE id=999",               "SELECT * FROM orders",                            -0.16, -0.14),
        ("gold empty, pred has rows",            "SELECT * FROM orders",                            "SELECT * FROM orders WHERE id=999",               -0.11, -0.09),
        ("both empty",                           "SELECT * FROM orders WHERE id=999",               "SELECT * FROM orders WHERE id=888",                0.99,  1.01),
        ("exact match",                          "SELECT id, customer, amount FROM orders",         "SELECT id, customer, amount FROM orders",          0.99,  1.01),
        ("alias mismatch — positional saves it", "SELECT COUNT(*) FROM orders",                     "SELECT COUNT(*) AS cnt FROM orders",               0.80,  1.01),
        ("partial rows (subset)",                "SELECT * FROM orders WHERE customer='alice'",     "SELECT * FROM orders",                            0.10,  0.85),
        ("partial cols",                         "SELECT id, customer FROM orders",                 "SELECT id, customer, amount FROM orders",          0.50,  0.91),
    ]

    all_pass = True
    for desc, pred, gold, lo, hi in cases:
        s = score(pred, gold)
        passed = lo <= s <= hi
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc:45s} → {s:+.4f}  (expect {lo:+.2f} to {hi:+.2f})")

    db_file.unlink(missing_ok=True)
    print("\nALL PASSED" if all_pass else "\nSOME TESTS FAILED")
