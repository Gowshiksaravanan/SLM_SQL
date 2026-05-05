# Evaluate SFT baseline on Spider dev

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

ROOT       = Path(__file__).parent.parent.parent
SPIDER_DIR = ROOT / "data" / "spider_official" / "spider_data"
DB_DIR     = SPIDER_DIR / "database"
DEV_FILE   = SPIDER_DIR / "dev.json"

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

sys.path.insert(0, str(Path(__file__).parent.parent))
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


NOCASE_RE = re.compile(r"((?:=|!=|<>)\s*'(?:[^']|'')*')(?!\s+COLLATE)", re.IGNORECASE)

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
    if gold_rows is None or pred_rows is None:
        return False
    if len(gold_rows) != len(pred_rows):
        return False
    gold_cols = {k.lower() for r in gold_rows for k in r}
    pred_cols = {k.lower() for r in pred_rows for k in r}
    if gold_cols == pred_cols:
        def row_tuple(r):
            return tuple(normalize(v) for _, v in sorted(r.items(), key=lambda x: x[0].lower()))
    else:
        def row_tuple(r):
            return tuple(sorted(normalize(v) for v in r.values()))
    return sorted(row_tuple(r) for r in gold_rows) == sorted(row_tuple(r) for r in pred_rows)


def load_model(checkpoint: str):
    print(f"Loading tokenizer from {cfg.MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.MODEL_PATH))
    tokenizer.padding_side = "left"
    tokenizer.chat_template = CHAT_TEMPLATE

    print(f"Loading Phi-4 (bf16) from {cfg.MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.MODEL_PATH),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print(f"Loading SFT baseline LoRA from {checkpoint} ...")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    print("Model ready.")
    return model, tokenizer


def predict(model, tokenizer, db_id: str, question: str) -> str:
    ddl = get_ddl(db_id)
    messages = [
        {"role": "system", "content": cfg.SYSTEM_PROMPT},
        {"role": "user",   "content": f"**Schema:**\n{ddl}\n\n**Question:** {question}"},
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
            max_new_tokens=512,   # SQL only — no CoT, so 512 is sufficient
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return extract_sql(response)


def extract_sql(response: str) -> str:
    # SFT baseline outputs plain SQL — strip markdown fences if present
    if "```sql" in response:
        return response.split("```sql")[1].split("```")[0].strip()
    if "```" in response:
        return response.split("```")[1].strip()
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="SFT baseline LoRA checkpoint dir")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of Spider dev examples to evaluate (default 100)")
    args = parser.parse_args()

    with open(DEV_FILE) as f:
        dev = json.load(f)

    examples = list(enumerate(dev[:args.n]))
    model, tokenizer = load_model(args.checkpoint)

    label = Path(args.checkpoint).name
    print(f"\nEvaluating [SFT baseline / {label}] on {len(examples)} Spider dev examples ...\n")
    print("=" * 80)

    correct, exec_errors, done = 0, 0, 0
    results = []

    for idx, ex in examples:
        done += 1
        pred_sql  = predict(model, tokenizer, ex["db_id"], ex["question"])
        gold_sql  = ex["query"]

        gold_rows, _ = run_sql(gold_sql, ex["db_id"])
        pred_rows, pred_err = run_sql(pred_sql, ex["db_id"])

        match = results_match(gold_rows, pred_rows)
        if match:
            correct += 1
        if pred_err:
            exec_errors += 1

        status = "✓" if match else "✗"
        print(f"\n[{idx:4d}] {status}  db={ex['db_id']}")
        print(f"       Q: {ex['question']}")
        print(f"    Gold: {gold_sql}")
        print(f"    Pred: {pred_sql}")
        if pred_err:
            print(f"    Exec Error: {pred_err}")

        if done % 10 == 0:
            print(f"\n  >> Running accuracy: {correct}/{done} = {correct/done:.1%}  "
                  f"(exec errors: {exec_errors})\n")

        results.append({
            "index":      idx,
            "db_id":      ex["db_id"],
            "question":   ex["question"],
            "gold_sql":   gold_sql,
            "pred_sql":   pred_sql,
            "match":      match,
            "pred_error": pred_err,
        })

    print("\n" + "=" * 80)
    print(f"FINAL: {correct}/{len(examples)} = {correct/len(examples):.1%} execution accuracy")
    print(f"       Execution errors: {exec_errors}/{len(examples)}")

    out_path = ROOT / "version3" / "sft_baseline" / f"eval_{label}_n{len(examples)}.json"
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint":    args.checkpoint,
            "n":             len(examples),
            "exec_accuracy": correct / len(examples),
            "exec_errors":   exec_errors,
            "results":       results,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
