import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config_stage2 as cfg

from rewards.utils import (completion_text,extract_sql,get_db_path,normalize_sql_for_execution,run_sql,)
from rewards.execution import compute_single
from rewards.format_reward import FORMAT_RE
from rewards.schema_reward import (hallucination_score,cot_consistency_score,)
from rewards.structural import (clause_coverage_score,complexity_score,reasoning_quality_score,)

proxy_max = (cfg.HALLUCINATION_WEIGHT+ cfg.COT_CONSISTENCY_WEIGHT+ cfg.CLAUSE_COVERAGE_WEIGHT
    + cfg.REASONING_WEIGHT
    + cfg.COMPLEXITY_WEIGHT
) # This weight will come into picture when there is no full execution reward
MAX_TOTAL = cfg.FORMAT_WEIGHT + cfg.EXECUTION_WEIGHT + proxy_max  # perfect score ceiling

call_counter = 0
LOG_EVERY = 50  # logging feature


def check_format(completion: str) -> float:# Returns FORMAT_WEIGHT if <think>...</think> + SQL structure is present, else 0
    return cfg.FORMAT_WEIGHT if FORMAT_RE.search(completion_text(completion)) else 0.0
def compute_exec(completion: str, gold_sql: str, db_id: str) -> float:# Returns a score in [-1, 1]: negative for errors, 0 for wrong rows, 1 for exact match
    pred_sql = extract_sql(completion)
    db_path = get_db_path(db_id)
    if db_path is None:
        return 0.0
    return compute_single(pred_sql, gold_sql, db_path)


def compute_proxies(completion: str, gold_sql: str, ddl: str) -> float:# Sum of all structural/semantic proxy signals — only used when execution isn't perfect
    text = completion_text(completion)
    pred_sql = extract_sql(completion)
    cot = text.split("</think>")[0].replace("<think>", "").strip() if "</think>" in text else ""
    return (
        hallucination_score(text, ddl)
        + cot_consistency_score(text)
        + clause_coverage_score(pred_sql, gold_sql)
        + complexity_score(pred_sql, gold_sql)
        + reasoning_quality_score(cot)
    )
# Three-tier cascading reward:
# Gate 1 — format check: wrong format → 0, stop
# Gate 2 — execution:    perfect match → MAX_TOTAL, stop
# Gate 3 — proxy blend:  partial execution + structural signals guide recovery
def combined_reward_func(completions: list,prompts: list,sql: list[str],db_id: list[str],ddl: list[str],**kwargs,) -> list[float]:
    global call_counter
    call_counter += 1
    log_this = (call_counter % LOG_EVERY == 0)
    scores = []
    for i, (completion, gold_sql, db, schema) in enumerate(
        zip(completions, sql, db_id, ddl)
    ):
#Gate1: Format check
        fmt = check_format(completion)
        if fmt == 0.0:
            scores.append(0.0)
            if log_this and i == 0:
                print(f"[reward] fmt=FAIL exec=- proxy=- total=0.0")
            continue
#Gate2: Execution reward
        exec_score = compute_exec(completion, gold_sql, db)
        exec_norm  = max(0.0, min(1.0, exec_score))
        if exec_score < 0.0:
            total = fmt + exec_score * cfg.EXECUTION_WEIGHT
            scores.append(total)
            if log_this and i == 0:
                print(f"[reward] fmt={fmt:.2f} exec={exec_score:.3f} proxy=SKIP total={total:.3f}")
            continue

        if exec_score >= 1.0: # If the execution reward is full no need of proxy reward
            scores.append(MAX_TOTAL)
            if log_this and i == 0:
                print(f"[reward] fmt={fmt:.2f} exec=PERFECT proxy=AUTO total={MAX_TOTAL:.3f}")
            continue

        actual_proxy = compute_proxies(completion, gold_sql, schema)
        proxy_contribution = exec_norm * proxy_max + (1.0 - exec_norm) * actual_proxy
        total = fmt + exec_score * cfg.EXECUTION_WEIGHT + proxy_contribution
        scores.append(total)

        if log_this and i == 0:
            print(
                f"[reward] fmt={fmt:.2f} exec={exec_score:.3f} "
                f"proxy={actual_proxy:.3f}→{proxy_contribution:.3f} total={total:.3f}"
            )

    return scores
