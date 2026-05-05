#Import the libraries
import argparse
import json
import re

import sqlite3
import sys
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "version3"))
import config_stage2 as cfg

#Configurations
STAGE2_JSONL  = ROOT / "version3" / "data" / "stage2_grpo.jsonl"
SYNSQL_JSONL  = ROOT / "pre-processing" / "stage2" / "data" / "synsql_stage2.jsonl"
EMPTY_DB_FILE = ROOT / "pre-processing" / "database_analysis" / "data" / "empty_databases.json"
OUTPUT_FILE   = ROOT / "version3" / "data" / "stage2_grpo.jsonl"
SHARD_DIR     = ROOT / "version3" / "data" / "filter_shards"

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


def load_model(): # Function for loading model
    print(f"Loading the tokenizer from {cfg.STAGE1B_MERGED} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.STAGE1B_MERGED))
    tokenizer.padding_side = "right"   # right-padding for loss computation
    tokenizer.chat_template = CHAT_TEMPLATE

    print(f"Loading the Stage 1 model")
    model = AutoModelForCausalLM.from_pretrained(str(cfg.STAGE1B_MERGED),dtype=torch.bfloat16,device_map="auto",trust_remote_code=True,attn_implementation="flash_attention_2",)
    model.eval()
    return model, tokenizer


def compute_loss_batch(model, tokenizer, records: list, device) -> list[float]:
    # Building full sequences
    full_texts = []
    sql_texts  = []
    for rec in records:
        prompt_text = tokenizer.applyCHAT_TEMPLATE(rec["prompt"], tokenize=False, add_generation_prompt=True)
       # assistant part
        sql_text = rec["sql"] + "<|im_end|>"
        full_texts.append(prompt_text + sql_text)
        sql_texts.append(sql_text)

    # Tokenize full sequences
    full_enc = tokenizer(full_texts,return_tensors="pt",padding=True,truncation=True,max_length=4096,
    ).to(device)

    # Tokenize SQL-only individually to get per-record SQL token lengths
    sql_lens = [
        len(tokenizer(t, add_special_tokens=False)["input_ids"])
        for t in sql_texts
    ]

    with torch.inference_mode():
        outputs = model(**full_enc)
    logits = outputs.logits 

    losses = []
    for i, rec in enumerate(records):
        input_ids = full_enc["input_ids"][i]        # (seq_len,)
        attn_mask = full_enc["attention_mask"][i]   # (seq_len,)
        sql_len   = sql_lens[i]

        # Real sequence length without padding
        real_len = attn_mask.sum().item()
        # SQL tokens are the last sql_len tokens of the real sequence
        sql_start = int(real_len) - sql_len

        if sql_start <= 0:
            losses.append(0.0)
            continue

        # Shift: logits at position t predict token at t+1
        # Loss on SQL tokens: positions sql_start-1 .. real_len-2 in logits
        lm_logits = logits[i, sql_start - 1 : real_len - 1, :]   # (sql_len, vocab)
        lm_labels = input_ids[sql_start : real_len]               # (sql_len,)

        loss = F.cross_entropy(lm_logits, lm_labels).item()
        losses.append(loss)

    return losses


def get_db_path(db_id: str) -> Path | None:
    for base in (cfg.SYNSQL_DB_DIR, cfg.SPIDER_DB_DIR, cfg.KAGGLE_DB_DIR):
        p = base / db_id / f"{db_id}.sqlite"
        if p.exists():
            return p
    return None


def run_sql_validate(sql: str, db_path: Path, timeout: int = 5) -> bool:
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


def pull_set_op_records(empty_db_ids: set) -> list:
    records = []
    _path_cache = {}
    _ddl_cache  = {}

    print(f"\nPulling EXCEPT + INTERSECT from {SYNSQL_JSONL} ...")
    with open(SYNSQL_JSONL) as f:
        for line in f:
            row = json.loads(line)
            if row.get("sql_complexity") != "Highly Complex":
                continue
            sql_upper = row["sql"].upper()
            if not (re.search(r'\bEXCEPT\b', sql_upper) or re.search(r'\bINTERSECT\b', sql_upper)):
                continue
            db_id = row["db_id"]
            if db_id in empty_db_ids:
                continue
            if db_id not in _path_cache:
                _path_cache[db_id] = get_db_path(db_id)
            db_path = _path_cache[db_id]
            if db_path is None:
                continue
            if db_id not in _ddl_cache:
                try:
                    con = sqlite3.connect(str(db_path))
                    rows = con.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
                    ).fetchall()
                    con.close()
                    _ddl_cache[db_id] = "\n\n".join(r[0] for r in rows)
                except Exception:
                    _ddl_cache[db_id] = ""
            ddl = _ddl_cache[db_id]
            if not ddl:
                continue
            if not run_sql_validate(row["sql"], db_path):
                continue
            question  = row["question"]
            knowledge = row.get("external_knowledge", "").strip()
            user_content = f"**Schema:**\n{ddl}\n\n**Question:** {question}"
            if knowledge:
                user_content += f"\n\n**Additional context:** {knowledge}"
            records.append({
                "prompt": [
                    {"role": "system", "content": cfg.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                "sql":   row["sql"],
                "db_id": db_id,
                "ddl":   ddl,
            })

    except_count    = sum(1 for r in records if re.search(r'\bEXCEPT\b',    r["sql"].upper()))
    intersect_count = sum(1 for r in records if re.search(r'\bINTERSECT\b', r["sql"].upper()))
    print(f"  EXCEPT: {except_count}  INTERSECT: {intersect_count}  total: {len(records)}")
    return records


def run_shard(rank: int, world_size: int, batch_size: int, percentile: float, dry_run: bool):
    print(f"\n[Rank {rank}/{world_size}] Starting loss-based filtering ...")

    records_50k = []
    with open(STAGE2_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                records_50k.append(json.loads(line))

    shard = records_50k[rank::world_size]
    print(f"[Rank {rank}] Processing {len(shard):,} / {len(records_50k):,} records")

    model, tokenizer = load_model()
    device = next(model.parameters()).device

    results = []  # list of (loss, record)
    total = len(shard)

    for start in range(0, total, batch_size):
        batch = shard[start : start + batch_size]
        losses = compute_loss_batch(model, tokenizer, batch, device)
        for rec, loss in zip(batch, losses):
            results.append((loss, rec))

        done = min(start + batch_size, total)
        if done % 500 == 0 or done == total:
            recent_losses = [l for l, _ in results[-500:]]
            avg = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            print(f"[Rank {rank}] {done:>6}/{total}  avg_loss={avg:.3f}")

    # Sort by loss descending, keep top `percentile`%
    results.sort(key=lambda x: -x[0])
    keep_n = int(len(results) * percentile)
    hard = [rec for _, rec in results[:keep_n]]

    loss_values = [l for l, _ in results]
    print(f"[Rank {rank}] Done — loss min={min(loss_values):.3f} max={max(loss_values):.3f} "
          f"median={sorted(loss_values)[len(loss_values)//2]:.3f}")
    print(f"[Rank {rank}] Keeping top {percentile*100:.0f}%: {len(hard):,} hard records")

    if not dry_run:
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        shard_file = SHARD_DIR / f"shard_{rank}_of_{world_size}.jsonl"
        with open(shard_file, "w") as f:
            for loss, rec in results[:keep_n]:
                entry = dict(rec)
                entry["_loss"] = loss
                f.write(json.dumps(entry) + "\n")
        print(f"[Rank {rank}] Wrote {shard_file}")


def merge(world_size: int, seed: int):
    """Merge all shard files + set-op records → stage2_grpo.jsonl."""
    import random
    random.seed(seed)

    with open(EMPTY_DB_FILE) as f:
        empty_data = json.load(f)
    db_list = empty_data if isinstance(empty_data, list) else empty_data.get("empty_databases", [])
    empty_db_ids = {e.get("db_id") or e.get("db_name") for e in db_list}

    hard_records = []
    for rank in range(world_size):
        shard_file = SHARD_DIR / f"shard_{rank}_of_{world_size}.jsonl"
        if not shard_file.exists():
            print(f"WARNING: {shard_file} not found — skipping")
            continue
        with open(shard_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec.pop("_loss", None)  # remove internal field
                    hard_records.append(rec)
        print(f"  Loaded shard {rank}: {len(hard_records):,} total so far")

    set_op_records = pull_set_op_records(empty_db_ids)

    # Deduplicate by (db_id, sql)
    seen: set = set()
    final = []
    def add(rec):
        key = (rec["db_id"], rec["sql"].strip())
        if key not in seen:
            seen.add(key)
            final.append(rec)

    for r in hard_records:
        add(r)
    for r in set_op_records:
        add(r)

    random.shuffle(final)

    print(f"\nFinal dataset: {len(final):,} records")
    print(f"  Hard (high loss) : {len(hard_records):,}")
    print(f"  Set-op top-up    : {len(set_op_records):,}")
    print(f"  After dedup      : {len(final):,}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for rec in final:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",        type=int,   default=0)
    parser.add_argument("--world-size",  type=int,   default=1)
    parser.add_argument("--batch-size",  type=int,   default=8,
                        help="Batch size for forward pass (higher = faster, loss-only is memory-light)")
    parser.add_argument("--percentile",  type=float, default=0.5,
                        help="Top fraction of hardest records to keep (default 0.5 = top 50%%)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--merge",       action="store_true",
                        help="Merge shard files into final output (no GPU needed)")
    args = parser.parse_args()

    if args.merge:
        merge(args.world_size, args.seed)
    else:
        run_shard(args.rank, args.world_size, args.batch_size, args.percentile, args.dry_run)


if __name__ == "__main__":
    main()
