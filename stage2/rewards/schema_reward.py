import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config_stage2 as cfg
from rewards.utils import (extract_cot,extract_sql,extract_tables_cols,parse_schema_entities,)


def schema_hallucination_reward_func(completions: list[str],prompts: list[str],**kwargs,) -> list[float]:
    ddls = kwargs["ddl"]
    rewards = []
    for completion, ddl in zip(completions, ddls):
        reward = hallucination_score(completion, ddl) * cfg.HALLUCINATION_WEIGHT
        rewards.append(reward)
    return rewards


# Checks what fraction of tables/columns in predicted SQL actually exist in the DDL
# Returns average of table precision and column precision — 1.0 means no hallucinations
def hallucination_score(completion: str, ddl: str) -> float:
    valid_tables, valid_cols = parse_schema_entities(ddl)
    if not valid_tables:
        return 0.0

    pred_sql = extract_sql(completion)
    pred_tables, pred_cols = extract_tables_cols(pred_sql)

    table_score = (
        len(pred_tables & valid_tables) / max(len(pred_tables), 1)
        if pred_tables else 1.0  # SELECT * with no explicit table ref — no hallucination penalty
    )
    col_score = (
        len(pred_cols & valid_cols) / max(len(pred_cols), 1)
        if pred_cols else 1.0
    )
    return (table_score + col_score) / 2


def cot_sql_consistency_reward_func(completions: list[str],prompts: list[str],**kwargs,) -> list[float]:
    rewards = []
    for completion in completions:
        reward = cot_consistency_score(completion) * cfg.COT_CONSISTENCY_WEIGHT
        rewards.append(reward)
    return rewards


# Checks that tables and columns used in the final SQL were actually mentioned in the reasoning
# Penalizes outputs where the <think> block is disconnected from the generated SQL
def cot_consistency_score(completion: str) -> float:
    cot = extract_cot(completion)
    if not cot:
        return 0.0

    pred_sql = extract_sql(completion)
    sql_tables, sql_cols = extract_tables_cols(pred_sql)

    if not sql_tables and not sql_cols:
        return 0.0

    cot_lower = cot.lower()

    table_mention = (
        sum(1 for t in sql_tables if t in cot_lower) / max(len(sql_tables), 1)
        if sql_tables else 1.0
    )
    col_mention = (
        sum(1 for c in sql_cols if c in cot_lower) / max(len(sql_cols), 1)
        if sql_cols else 1.0
    )

    return (table_mention + col_mention) / 2


if __name__ == "__main__": #For unit test
    DDL = """
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer TEXT,
        amount REAL
    );
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT,
        price REAL
    );
    """

    def fmt_str(score: float, weight: float) -> str:
        return f"raw={score/weight:.3f}  weighted={score:.3f}"

    print("schema_hallucination_reward tests:")
    hallucination_cases = [
        ("correct tables+cols",
         "<think>use orders table, select customer and amount</think>\nSELECT customer, amount FROM orders",
         1.0, True),
        ("hallucinated table (SELECT * so no col hallucination)",
         "<think>I need invoices</think>\nSELECT * FROM invoices",
         0.5, True),  # table=0.0, cols empty (SELECT *) → 1.0, avg=0.5
        ("hallucinated column",
         "<think>use orders</think>\nSELECT ghost_col FROM orders",
         0.5, True),  # table OK (1.0), col wrong (0.0) → 0.5
        ("mixed: real table + fake col",
         "<think>use orders</think>\nSELECT id, fake_field FROM orders",
         0.75, True),  # table=1.0, col=0.5 → 0.75
        ("empty ddl fallback",
         "<think>use orders</think>\nSELECT id FROM orders",
         0.0, False),  # ddl="" → can't evaluate
    ]

    all_pass = True
    for desc, completion, expected_raw, use_ddl in hallucination_cases:
        ddl_arg = DDL if use_ddl else ""
        result = hallucination_score(completion, ddl_arg)
        passed = abs(result - expected_raw) < 0.05
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc:40s} → raw={result:.3f}  (expect ~{expected_raw:.3f})")

    print("\ncot_sql_consistency_reward tests:")
    consistency_cases = [
        ("all entities mentioned in cot",
         "<think>use orders table, look at customer and amount columns</think>\nSELECT customer, amount FROM orders",
         1.0),
        ("no think block",
         "SELECT customer FROM orders",
         0.0),
        ("entities not mentioned in cot",
         "<think>I need to write a query</think>\nSELECT customer, amount FROM orders",
         0.0),
        ("partial mention — table yes, col no",
         "<think>the orders table has what I need</think>\nSELECT customer FROM orders",
         0.5),  # table=1.0, col=0.0 → 0.5
    ]

    for desc, completion, expected_raw in consistency_cases:
        result = cot_consistency_score(completion)
        passed = abs(result - expected_raw) < 0.05
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc:45s} → raw={result:.3f}  (expect ~{expected_raw:.3f})")

    print("\nALL PASSED" if all_pass else "\nSOME TESTS FAILED")
