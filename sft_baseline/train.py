# SFT baseline: schema + question -> SQL, no CoT reasoning

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg


# Reads pre-processed Kaggle/Wide JSONL files that already have a ddl field.
# Unlike raw SynSQL, these records carry the schema inline — no SQLite lookup needed.
def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("sql", "").strip() and r.get("ddl", "").strip():
                records.append(r)
    return records


def build_dataset(debug: bool = False) -> tuple[Dataset, Dataset]:
    kaggle  = load_records(cfg.KAGGLE_FILE)
    wide    = load_records(cfg.WIDE_FILE)
    records = kaggle + wide

    print(f"Kaggle loaded  : {len(kaggle):,}")
    print(f"Wide loaded    : {len(wide):,}")
    print(f"Combined total : {len(records):,}")

    random.seed(cfg.SEED)
    random.shuffle(records)

    if debug:
        records = records[:256]
        print("DEBUG: using 256 records")
    else:
        records = records[:cfg.MAX_TRAIN_SAMPLES]
        print(f"Capped to      : {len(records):,} samples")

    cut        = int(len(records) * (1 - cfg.VAL_SPLIT))
    train_rows = [{"messages": make_messages(r)} for r in records[:cut]]
    val_rows   = [{"messages": make_messages(r)} for r in records[cut:]]
    train = Dataset.from_list(train_rows)
    val   = Dataset.from_list(val_rows)
    print(f"Train          : {len(train):,}")
    print(f"Val            : {len(val):,}")
    return train, val


def make_user_content(r: dict) -> str:
    ext = r.get("external_knowledge", "").strip()
    msg = f"**Schema:**\n{r['ddl']}\n\n**Question:** {r['question']}"
    if ext:
        msg += f"\n\n**External Knowledge:** {ext}"
    return msg


def make_assistant_content(r: dict) -> str:
    # Plain SQL only — no CoT, no <think> tags
    return r['sql'].rstrip().rstrip(';').rstrip()


def make_messages(r: dict) -> list:
    return [
        {"role": "system",    "content": cfg.SYSTEM_PROMPT},
        {"role": "user",      "content": make_user_content(r)},
        {"role": "assistant", "content": make_assistant_content(r)},
    ]


# {% generation %} marks the assistant response boundaries so SFTTrainer's
# assistant_only_loss=True can mask out the prompt tokens and compute loss only on SQL output.
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


def load_model_and_tokenizer():
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
    tokenizer.padding_side = "right"  # right-pad for SFT (left-pad is for generation)
    tokenizer.chat_template = CHAT_TEMPLATE

    print("Loading Phi-4 (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_PATH,
        dtype=torch.bfloat16,
        device_map=None,  # DDP handles device placement — don't let HF auto-shard
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Smoke test: 256 records, 2 steps, no checkpoint save")
    args = parser.parse_args()

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_dataset(debug=args.debug)
    model, tokenizer = load_model_and_tokenizer()

    sft_cfg = SFTConfig(
        output_dir=str(cfg.OUTPUT_DIR),

        num_train_epochs=1             if args.debug else cfg.NUM_EPOCHS,
        max_steps=2                    if args.debug else -1,

        per_device_train_batch_size=1  if args.debug else cfg.BATCH_GPU,
        per_device_eval_batch_size=1   if args.debug else cfg.BATCH_GPU,
        gradient_accumulation_steps=1  if args.debug else cfg.GRAD_ACCUM,

        max_length=cfg.MAX_SEQ_LEN,

        learning_rate=cfg.LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.0               if args.debug else cfg.WARMUP_RATIO,
        weight_decay=cfg.WEIGHT_DECAY,
        optim="adamw_torch_fused",

        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        eval_strategy="steps",
        eval_steps=1                   if args.debug else cfg.EVAL_STEPS,
        save_strategy="steps",
        save_steps=9999                if args.debug else cfg.SAVE_STEPS,
        save_total_limit=cfg.SAVE_TOTAL,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        assistant_only_loss=True,
        ddp_find_unused_parameters=False,

        logging_steps=cfg.LOGGING_STEPS,
        report_to=cfg.REPORT_TO,

        seed=cfg.SEED,
        data_seed=cfg.SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("\nStarting SFT baseline training ...")
    trainer.train()

    if not args.debug:
        save_path = cfg.OUTPUT_DIR / "final"
        print(f"\nSaving LoRA adapter to {save_path} ...")
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print("Done.")


if __name__ == "__main__":
    main()
