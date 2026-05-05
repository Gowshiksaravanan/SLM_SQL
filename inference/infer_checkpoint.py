# Evaluate Stage 1 checkpoint on Spider dev

import argparse
import json
import sqlite3
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "phi-4"
SPIDER_DIR = ROOT / "data" / "spider_official" / "spider_data"
DB_DIR     = SPIDER_DIR / "database"
DEV_FILE   = SPIDER_DIR / "dev.json"

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a natural language question, "
    "think step-by-step inside <think>...</think> tags to identify the relevant tables "
    "and columns, reason through the query logic, then write the final SQLite SQL query."
)

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
    # Fallback: extract from SQLite file directly
    db_file = DB_DIR / db_id / f"{db_id}.sqlite"
    if not db_file.exists():
        return ""
    con = sqlite3.connect(db_file)
    rows = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    con.close()
    return "\n\n".join(r[0] for r in rows)


def load_model(checkpoint: str):
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    tokenizer.chat_template = CHAT_TEMPLATE

    print("Loading base model (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if checkpoint:
        print(f"Loading LoRA adapter from {checkpoint} ...")
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        print("Running base model (no LoRA adapter)")
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, db_id: str, question: str, max_new_tokens: int = 2048) -> str:
    ddl = get_ddl(db_id)
    user_content = f"**Schema:**\n{ddl}\n\n**Question:** {question}"
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
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
    """Pull the SQL that comes after </think>."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint dir; omit for base model")
    parser.add_argument("--n", type=int, default=10, help="Number of dev examples to run")
    parser.add_argument("--indices", type=int, nargs="+", help="Specific dev indices to run")
    args = parser.parse_args()

    with open(DEV_FILE) as f:
        dev = json.load(f)

    if args.indices:
        examples = [(i, dev[i]) for i in args.indices]
    else:
        examples = list(enumerate(dev[:args.n]))

    model, tokenizer = load_model(args.checkpoint)

    print(f"\nRunning inference on {len(examples)} Spider dev examples ...\n")
    print("=" * 80)

    for i, ex in examples:
        response = predict(model, tokenizer, ex["db_id"], ex["question"])
        pred_sql = extract_sql(response)

        print(f"[{i+1}] db_id    : {ex['db_id']}")
        print(f"     question : {ex['question']}")
        print(f"     gold SQL : {ex['query']}")
        print(f"     pred SQL : {pred_sql}")

        think_part = response.split("</think>")[0].replace("<think>", "").strip() if "<think>" in response else ""
        if think_part:
            print(f"     think[:200]: {think_part[:200]}")
        print()


if __name__ == "__main__":
    main()
