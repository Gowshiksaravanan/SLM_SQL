import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config_stage2 as cfg
from rewards.utils import extract_cot, extract_sql

# These 8 clauses are the ones the model most commonly gets wrong or omits.Simple SELECT/FROM/WHERE are nearly universal, so matching them adds no signal.Set operations (UNION/INTERSECT/EXCEPT) and aggregation modifiers (GROUP BY/HAVING)
# are the real failure modes — the model learns CTE rewrites that skip them entirely.
CLAUSE_PATTERNS: dict[str, re.Pattern] = {
    "UNION": re.compile(r"\bUNION\b", re.IGNORECASE),
    "INTERSECT": re.compile(r"\bINTERSECT\b", re.IGNORECASE),
    "EXCEPT": re.compile(r"\bEXCEPT\b", re.IGNORECASE),
    "GROUP BY": re.compile(r"\bGROUP\s+BY\b",re.IGNORECASE),
    "HAVING":re.compile(r"\bHAVING\b",re.IGNORECASE),
    "ORDER BY":re.compile(r"\bORDER\s+BY\b",re.IGNORECASE),
    "LIMIT":re.compile(r"\bLIMIT\b", re.IGNORECASE),
    "WITH":re.compile(r"\bWITH\b", re.IGNORECASE),
}

def clause_set(sql: str) -> set[str]:
    return {name for name, pat in CLAUSE_PATTERNS.items() if pat.search(sql)}


def clause_coverage_reward_func(completions: list[str],prompts: list[str],**kwargs,) -> list[float]:
    gold_sqls = kwargs["sql"]

    rewards = []
    for completion, gold_sql in zip(completions, gold_sqls):
        pred_sql = extract_sql(completion)
        reward = clause_coverage_score(pred_sql, gold_sql) * cfg.CLAUSE_COVERAGE_WEIGHT
        rewards.append(reward)
    return rewards


# F1 (not just recall) so the model isn't rewarded for dumping every clause into every query.
# Recall alone would give full score to a query that has UNION + INTERSECT + HAVING + GROUP BY
# even if gold is a simple SELECT. Precision penalizes that over-generation.
def clause_coverage_score(pred_sql: str, gold_sql: str) -> float:
    pred_clauses = clause_set(pred_sql)
    gold_clauses = clause_set(gold_sql)

    if not gold_clauses and not pred_clauses:
        return 1.0  # neither uses special clauses — simple query, full credit

    intersection = pred_clauses & gold_clauses
    precision = len(intersection) / max(len(pred_clauses), 1)
    recall    = len(intersection) / max(len(gold_clauses), 1)

    if precision + recall == 0:
        return 0.0
    return (precision + recall) / 2


def complexity_reward_func(
    completions: list[str],
    prompts: list[str],
    **kwargs,
) -> list[float]:
    gold_sqls = kwargs["sql"]

    rewards = []
    for completion, gold_sql in zip(completions, gold_sqls):
        pred_sql = extract_sql(completion)
        reward = complexity_score(pred_sql, gold_sql) * cfg.COMPLEXITY_WEIGHT
        rewards.append(reward)
    return rewards


# Gaussian on log token-count ratio — peaks at 1.0 when pred and gold are same length,
# drops symmetrically for both over-simplified and over-engineered queries.
# Log ratio keeps the penalty proportional: 2x too long = same score as 2x too short.
def complexity_score(pred_sql: str, gold_sql: str) -> float:
    pred_tokens = max(len(pred_sql.split()), 1)
    gold_tokens = max(len(gold_sql.split()), 1)
    log_ratio = math.log(pred_tokens / gold_tokens)
    return math.exp(-0.5 * log_ratio ** 2)
SQL_TERMS = {
    "select", "from", "where", "join", "group", "order", "having",
    "union", "intersect", "except", "count", "sum", "avg", "max", "min",
    "distinct", "limit", "inner", "outer", "left", "right", "on", "as",
    "with", "subquery", "nested", "filter", "aggregate", "condition",
}

STEP_MARKERS = re.compile(
    r"\b(step|first|second|third|next|then|finally|identify|select|join|filter"
    r"|group|sort|compute|calculate|find|determine|need|must|should)\b",
    re.IGNORECASE,
)

def reasoning_quality_reward_func(
    completions: list[str],
    prompts: list[str],
    **kwargs,
) -> list[float]:
    rewards = []
    for completion in completions:
        cot = extract_cot(completion)
        reward = reasoning_quality_score(cot) * cfg.REASONING_WEIGHT
        rewards.append(reward)
    return rewards


# 5 proxy signals — none is reliable alone, but together they filter out empty/garbage CoT.
# Stage 1B already trains CoT format, so this weight is low (0.4). The goal here is just
# to penalize outputs where <think> is a one-liner with no actual reasoning.
def reasoning_quality_score(cot: str) -> float:
    if not cot:
        return 0.0

    score = 0.0
    words = cot.lower().split()
    word_set = set(words)

    # Sub-signal 1: length ≥50 words
    if len(words) >= 50:
        score += 0.20

    # Sub-signal 2: SQL term density (0.03 per unique term, max 0.20)
    sql_hits = word_set & SQL_TERMS
    score += min(len(sql_hits) * 0.03, 0.20)

    # Sub-signal 3: structure — ≥3 non-empty lines
    non_empty_lines = [l for l in cot.splitlines() if l.strip()]
    if len(non_empty_lines) >= 3:
        score += 0.15

    # Sub-signal 4: step/transition markers
    if STEP_MARKERS.search(cot):
        score += 0.15

    # Sub-signal 5: schema entity mentions — unique words that look like identifiers
    # (length ≥4, contains underscore or mixed case → likely a schema name)
    schema_refs = sum(
        1 for w in word_set
        if len(w) >= 4 and ("_" in w or (w != w.lower() and w != w.upper()))
    )
    score += min(schema_refs * 0.05, 0.30)

    return min(score, 1.0)


if __name__ == "__main__":
    print("clause_coverage_reward tests:")
    clause_cases = [
        ("both simple — no special clauses",
         "SELECT id FROM orders",       "SELECT * FROM orders",           0.95, 1.05),
        ("pred missing INTERSECT",
         "SELECT id FROM orders",       "SELECT id FROM orders INTERSECT SELECT id FROM items", 0.0, 0.05),
        ("pred has GROUP BY, gold too",
         "SELECT customer, COUNT(*) FROM orders GROUP BY customer",
         "SELECT customer, COUNT(*) FROM orders GROUP BY customer",       0.95, 1.05),
        ("pred has GROUP BY + HAVING, gold just GROUP BY",
         "SELECT customer, COUNT(*) FROM orders GROUP BY customer HAVING COUNT(*) > 2",
         "SELECT customer, COUNT(*) FROM orders GROUP BY customer",       0.60, 0.85),
    ]
    all_pass = True
    for desc, pred, gold, lo, hi in clause_cases:
        s = clause_coverage_score(pred, gold)
        passed = lo <= s <= hi
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc:55s} → {s:.3f}  (expect {lo:.2f}–{hi:.2f})")

    print("\ncomplexity_reward tests:")
    complexity_cases = [
        ("identical length",        "SELECT id FROM orders",           "SELECT id FROM orders",  0.95, 1.05),
        ("pred 3x longer (12 vs 4 tokens)",
                                    "SELECT id FROM orders WHERE id > 0 AND id < 100",
                                    "SELECT id FROM orders",           0.50, 0.60),
        ("pred 3x shorter (4 vs 12 tokens)",
                                    "SELECT id FROM orders",
                                    "SELECT id FROM orders WHERE id > 0 AND id < 100", 0.50, 0.60),
    ]
    for desc, pred, gold, lo, hi in complexity_cases:
        s = complexity_score(pred, gold)
        passed = lo <= s <= hi
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc:45s} → {s:.3f}  (expect {lo:.2f}–{hi:.2f})")

    print("\nreasoning_quality_reward tests:")
    quality_cases = [
        ("no cot", "", 0.0, 0.05),
        ("minimal cot", "use orders table", 0.0, 0.25),
        ("good structured cot",
         "Step 1: Identify the relevant tables — orders contains customer data.\n"
         "Step 2: Filter by customer_id using WHERE clause.\n"
         "Step 3: Use COUNT(*) aggregate with GROUP BY to count orders per customer.\n"
         "Step 4: Apply HAVING filter to keep only customers with more than 5 orders.\n"
         "The final query joins orders and uses GROUP BY customer_id, HAVING COUNT > 5.",
         0.55, 1.05),
    ]
    for desc, cot, lo, hi in quality_cases:
        s = reasoning_quality_score(cot)
        passed = lo <= s <= hi
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc:45s} → {s:.3f}  (expect {lo:.2f}–{hi:.2f})")

    print("\nALL PASSED" if all_pass else "\nSOME TESTS FAILED")
