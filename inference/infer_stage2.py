# Evaluate final Stage 2 model on Spider dev

import argparse
import json
import re
import sqlite3
import sys
import threading
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT       = Path(__file__).parent.parent
SPIDER_DIR = ROOT / "data" / "spider_official" / "spider_data"
DB_DIR     = SPIDER_DIR / "database"
DEV_FILE   = SPIDER_DIR / "dev.json"

sys.path.insert(0, str(Path(__file__).parent))
import config_stage2 as cfg
from rewards.utils import normalize_sql_for_execution

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant<|im_sep|>' }}"
    "{% generation %}"
    "{{ message['content'] + '<|im_end|>' }}"
    "{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"
)


def get_ddl(db_id: str) -> str:
    schema_file = DB_DIR / db_id / "schema.sql"
    if schema_file.exists():
        lines = [
            l for l in schema_file.read_text().splitlines()
            if not l.strip().lower().startswith(("drop", "pragma", "insert"))
        ]
        return "\n".join(lines).strip()
    db_file = DB_DIR / db_id / f"{db_id}.sqlite"
    if not db_file.exists():
        return ""
    con = sqlite3.connect(str(db_file))
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    con.close()
    return "\n\n".join(r[0] for r in rows)


NOCASE_RE = re.compile(
    r"""((?:=|!=|<>)\s*(?:'(?:[^']|'')*'|"(?:[^"]|"")*"))(?!\s+COLLATE)""",
    re.IGNORECASE,
)

def add_nocase_collation(sql: str) -> str:
    return NOCASE_RE.sub(r"\1 COLLATE NOCASE", sql)


def run_sql(sql: str, db_id: str, timeout: int = 10):
    db_file = DB_DIR / db_id / f"{db_id}.sqlite"
    if not db_file.exists():
        return None, "db not found"
    sql = add_nocase_collation(normalize_sql_for_execution(sql))
    result = [None, None]
    def run_query():
        try:
            con = sqlite3.connect(str(db_file))
            con.row_factory = sqlite3.Row
            rows = con.execute(sql).fetchall()
            con.close()
            result[0] = [dict(r) for r in rows]
        except Exception as e:
            result[1] = str(e)
    t = threading.Thread(target=run_query, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, f"execution timeout ({timeout}s)"
    return result[0], result[1]


def normalize(v):
    if v is None:
        return ""
    try:
        return str(round(float(v), 4))
    except (ValueError, TypeError):
        return str(v).lower().strip()


def results_match(gold_rows, pred_rows) -> bool:
    """Order-insensitive and column-order-insensitive result-set comparison."""
    if gold_rows is None or pred_rows is None:
        return False
    if len(gold_rows) != len(pred_rows):
        return False

    gold_cols = {k.lower() for r in gold_rows for k in r}
    pred_cols = {k.lower() for r in pred_rows for k in r}

    # If pred has extra columns but contains all gold columns, project pred to gold columns
    if gold_cols and gold_cols.issubset(pred_cols) and gold_cols != pred_cols:
        pred_rows = [{k: v for k, v in r.items() if k.lower() in gold_cols} for r in pred_rows]
        pred_cols = gold_cols

    if gold_cols == pred_cols:
        # Same column names: sort values by column name to neutralize SELECT ordering
        def row_tuple(r):
            return tuple(normalize(v) for _, v in sorted(r.items(), key=lambda x: x[0].lower()))
    else:
        # Different aliases: compare value multisets only (ignores column names)
        def row_tuple(r):
            return tuple(sorted(normalize(v) for v in r.values()))

    return sorted(row_tuple(r) for r in gold_rows) == sorted(row_tuple(r) for r in pred_rows)


def load_model(checkpoint: str | None, model_path: str | None = None, use_stage2_merged: bool = False):
    if model_path:
        base_path = model_path
    elif use_stage2_merged:
        base_path = str(cfg.STAGE2_MERGED)
    else:
        base_path = str(cfg.STAGE1B_MERGED)
    print(f"Loading tokenizer from {base_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.padding_side = "left"
    tokenizer.chat_template = CHAT_TEMPLATE

    print(f"Loading model (bf16) from {base_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if checkpoint:
        print(f"Loading LoRA adapter from {checkpoint} ...")
        model = PeftModel.from_pretrained(model, checkpoint)
        print("LoRA loaded.")
    elif not model_path:
        print("No LoRA adapter — running base model.")

    model.eval()
    return model, tokenizer


def predict(model, tokenizer, db_id: str, question: str, max_new_tokens: int = 4096) -> str:
    ddl = get_ddl(db_id)
    user_content = f"**Schema:**\n{ddl}\n\n**Question:** {question}"
    messages = [
        {"role": "system", "content": cfg.SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids      = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_sql(response: str) -> str:
    if "</think>" in response:
        after = response.split("</think>")[-1].strip()
        # Strip markdown code fences if present
        if after.startswith("```"):
            after = after.split("```")[1]
            if after.lower().startswith("sql"):
                after = after[3:]
            return after.strip()
        return after
    # Fallback: look for ```sql block
    if "```sql" in response:
        return response.split("```sql")[1].split("```")[0].strip()
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Stage 2 LoRA checkpoint dir (e.g. version3/checkpoints/stage2/checkpoint-500)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run Stage 1B merged model without any Stage 2 LoRA")
    parser.add_argument("--model-path", default=None,
                        help="Override base model path (e.g. models/phi-4 for raw model)")
    parser.add_argument("--stage2-merged", action="store_true",
                        help="Use stage2/merged as base (for Spider GRPO checkpoints)")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of Spider dev examples to evaluate (default 100)")
    parser.add_argument("--indices", type=int, nargs="+",
                        help="Specific Spider dev indices to run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print reasoning trace for each example")
    args = parser.parse_args()

    checkpoint = None if (args.baseline or args.model_path) else args.checkpoint
    use_stage2_merged = args.stage2_merged

    with open(DEV_FILE) as f:
        dev = json.load(f)

    if args.indices:
        examples = [(i, dev[i]) for i in args.indices]
    else:
        examples = list(enumerate(dev[:args.n]))

    model, tokenizer = load_model(checkpoint, model_path=args.model_path, use_stage2_merged=use_stage2_merged)

    if args.model_path:
        label = Path(args.model_path).name
    elif checkpoint:
        label = Path(checkpoint).name
    else:
        label = "stage1b_baseline"
    print(f"\nEvaluating [{label}] on {len(examples)} Spider dev examples ...\n")
    print("=" * 80)

    correct = 0
    exec_errors = 0
    results = []

    for idx, (i, ex) in enumerate(examples):
        response   = predict(model, tokenizer, ex["db_id"], ex["question"])
        pred_sql   = extract_sql(response)
        gold_sql   = ex["query"]

        gold_rows, gold_err = run_sql(gold_sql, ex["db_id"])
        pred_rows, pred_err = run_sql(pred_sql, ex["db_id"])

        match = results_match(gold_rows, pred_rows)
        if match:
            correct += 1
        if pred_err:
            exec_errors += 1

        status = "✓" if match else "✗"
        print(f"[{i:4d}] {status}  db={ex['db_id']}")
        print(f"       Q: {ex['question']}")
        print(f"    Gold: {gold_sql}")
        print(f"    Pred: {pred_sql}")
        if pred_err:
            print(f"   Error: {pred_err}")
        if args.verbose and "<think>" in response:
            think = response.split("</think>")[0].replace("<think>", "").strip()
            print(f"   Think: {think[:300]}")
        print()

        results.append({
            "index": i,
            "db_id": ex["db_id"],
            "question": ex["question"],
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "match": match,
            "pred_error": pred_err,
        })

        # Running accuracy
        done = idx + 1
        print(f"  >> Running accuracy: {correct}/{done} = {correct/done:.1%}  "
              f"(exec errors: {exec_errors})")
        print()

    print("=" * 80)
    print(f"FINAL: {correct}/{len(examples)} = {correct/len(examples):.1%} execution accuracy")
    print(f"       Execution errors: {exec_errors}/{len(examples)}")

    # Save results
    out_file = Path(f"version3/logs/eval_{label}_n{len(examples)}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "checkpoint": str(checkpoint),
            "n": len(examples),
            "exec_accuracy": correct / len(examples),
            "exec_errors": exec_errors,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
