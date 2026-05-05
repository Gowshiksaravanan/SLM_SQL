# Stage 2 GRPO training -- Key file for the project

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SQLITE_TMPDIR", "/home/gsarava2/sqlite_tmp") # Workaround for not writing temp sqlite files in root of h200
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") # Workaround for not writing temp sqlite files in root of h200

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
import config_stage2 as cfg
from rewards.combined_reward import combined_reward_func

#This chat template i formatted with the help of AI
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


# Loading the GRPO training data — each record has prompt, sql, db_id, ddl fields
def load_dataset(debug: bool = False) -> tuple[Dataset, Dataset]:
    if not cfg.STAGE2_DATA.exists():
        raise FileNotFoundError(
            f"{cfg.STAGE2_DATA} not found. "
        
        )

    records = []
    with open(cfg.STAGE2_DATA) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records):,} records from {cfg.STAGE2_DATA}")

    random.seed(cfg.SEED)
    random.shuffle(records)

    if debug: # For debug purpose
        records = records[:64]

    # cap val at 500 so we don't waste too many examples on non-GRPO evaluation
    cut = max(int(len(records) * 0.98), len(records) - 500)
    train_rows = records[:cut]
    val_rows   = records[cut:]

    print(f"Train: {len(train_rows):,}  Val: {len(val_rows):,}")
    train_ds = Dataset.from_list(train_rows)
    val_ds   = Dataset.from_list(val_rows)
    return train_ds, val_ds


# Stage 2 starts from the merged Stage 1B weights — fresh LoRA added on top
# The base weights (From stage 1b) also act as the frozen reference policy for KL divergence
def load_model_and_tokenizer():
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.STAGE1B_MERGED))
    tokenizer.padding_side = "left"  # GRPO generates batches of completions — left-pad required

    print(f"Loading merged Stage 1B model {cfg.STAGE1B_MERGED}")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(cfg.STAGE1B_MERGED),
        dtype=torch.bfloat16,
        device_map=None,        
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_model.config.use_cache = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        )
    parser.add_argument("--resume_from_checkpoint", default=None,
                        )
    parser.add_argument("--data", default=None,
                        )
    parser.add_argument("--epochs", type=int, default=None,
                        )
    parser.add_argument("--output-dir", default=None,)
    args = parser.parse_args()

    if args.data:
        cfg.STAGE2_DATA = Path(args.data)
    if args.epochs:
        cfg.NUM_EPOCHS = args.epochs
    if args.output_dir:
        cfg.OUTPUT_DIR = Path(args.output_dir)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = load_dataset(debug=args.debug)
    model, tokenizer = load_model_and_tokenizer()

    grpo_cfg = GRPOConfig(
        output_dir=str(cfg.OUTPUT_DIR),
        # GRPO-specific
        num_generations=cfg.NUM_GENERATIONS,   # 8 rollouts per prompt
        beta=cfg.BETA,
        max_prompt_length=cfg.MAX_PROMPT_LENGTH,
        max_completion_length=cfg.MAX_COMPLETION_LENGTH,
        # Training
        per_device_train_batch_size=cfg.BATCH_GPU,
        gradient_accumulation_steps=cfg.GRAD_ACCUM,
        num_train_epochs=cfg.NUM_EPOCHS,
        learning_rate=cfg.LR,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.WARMUP_RATIO,
        weight_decay=0.01,
        # Precision / memory
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    # Eval & checkpointing
        eval_strategy="no",  # GRPO has no eval_loss — reward is the only training signal
        save_strategy="steps",
        save_steps=cfg.SAVE_STEPS,
        save_total_limit=cfg.SAVE_TOTAL,
        load_best_model_at_end=False,
        # Logging
        logging_steps=cfg.LOGGING_STEPS,
        report_to=cfg.REPORT_TO,
        seed=cfg.SEED,
        use_vllm=False,  # vLLM not installed (Environment issue with CUDA 3) — uses standard HF generate() for rollouts
        max_steps=2 if args.debug else -1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[combined_reward_func],
        args=grpo_cfg,
        train_dataset=train_ds,
    )

    print("Starting GRPO training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if not args.debug:
        final_dir = cfg.OUTPUT_DIR / "final"
        print(f"Saving final model to {final_dir}")
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

    print("over")


if __name__ == "__main__":
    main()
