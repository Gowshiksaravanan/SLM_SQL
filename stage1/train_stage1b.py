#This file is responsible for training 
import argparse      
import random        
import sqlite3
import ijson
import torch
from datasets import Dataset                          
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer                
import config as cfg

#-----------------------------------------------------------------

def get_ddl(db_id: str, cache: dict) -> str: #Function to return ddl of the table
    if db_id in cache: # If the db is already in cache then return it. Caching it so that we can save the training and inference time.
        return cache[db_id]

    db_file = cfg.SYNSQL_DB_DIR / db_id / f"{db_id}.sqlite"  # path to SQLite file
    if not db_file.exists():# A fallback to avoid error if some db doesnt have a sqlite file     
        cache[db_id] = ""
        return ""
#Hitting and retrieveing the data from DB
    con  = sqlite3.connect(str(db_file)) # DB connection
    rows = con.execute(                
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
    ).fetchall()
    con.close()
    ddl = "\n\n".join(r[0] for r in rows)# A formatting for DDL
    cache[db_id] = ddl                     
    return ddl




def load_synsql(limit: int | None = None) -> list: # To stream through synsql and get training samples
    records   = []
    ddl_cache = {} 

    with open(cfg.SYNSQL_FILE, "rb") as f:
        for rec_sample in ijson.items(f, "item"):    
            cot= rec_sample.get("cot", "").strip()
            sql =rec_sample.get("sql", "").strip() 
            if not cot or not sql:            
                continue

            db_id = rec_sample.get("db_id", "")
            ddl= get_ddl(db_id, ddl_cache) # load the schema from sqlite
            if not ddl:  # skip if empty                      
                continue
            records.append({
                "db_id":              db_id,
                "question":           rec_sample.get("question", ""),
                "external_knowledge": rec_sample.get("external_knowledge", ""),
                "cot":                cot,
                "sql":                sql,
                "ddl":                ddl,
            })

            if limit and len(records) >= limit:# debug purpose
                break
    return records

def build_dataset(debug: bool = False) -> tuple[Dataset, Dataset]:
    limit= 512 if debug else None
    records = load_synsql(limit=limit)
    if not records:
        raise RuntimeError("No records loaded from SynSQL-2.5M.")

    random.seed(cfg.SEED)
    random.shuffle(records)
    split_at= int(len(records) * (1 - cfg.VAL_SPLIT))# Splitting for val
    train_rows = [{"messages": make_messages(r)} for r in records[:split_at]]
    val_rows = [{"messages": make_messages(r)} for r in records[split_at:]]

    train = Dataset.from_list(train_rows)
    val= Dataset.from_list(val_rows)
    return train, val


def make_user_content(r: dict) -> str:
    ext= r.get("external_knowledge", "").strip()
    msg= f"**Schema:**\n{r['ddl']}\n\n**Question:** {r['question']}"
    if ext:
        msg += f"\n\n**External Knowledge:** {ext}"   
    return msg


def make_assistant_content(r: dict) -> str:
    sql = r['sql'].rstrip().rstrip(';').rstrip()      
    return f"<think>\n{r['cot']}\n</think>\n\n{sql}"  


def make_messages(r: dict) -> list:
    return [
        {"role": "system","content": cfg.SYSTEM_PROMPT},
        {"role": "user","content": make_user_content(r)},
        {"role": "assistant", "content": make_assistant_content(r)},
    ]

#-----------------------------------------------------------------------------
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant<|im_sep|>' }}"
    "{% generation %}"                    # TRL loss-masking directive — loss applied from here
    "{{ message['content'] + '<|im_end|>' }}"
    "{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_sep|>' }}{% endif %}"
)


def load_model_and_tokenizer():
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)
    tokenizer.padding_side= "right" #right pad during the training and left padding at the inference
    tokenizer.chat_template = CHAT_TEMPLATE  # override saved template with training template

    print("Loading Phi-4 (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_PATH,
        dtype=torch.bfloat16,   
        device_map=None,       
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # faster attention kernel
    )
    model.config.use_cache = False   

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
    parser.add_argument("--debug", action="store_true",)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    args = parser.parse_args()
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_dataset(debug=args.debug)
    model, tokenizer = load_model_and_tokenizer()

    sft_cfg = SFTConfig(
        output_dir=str(cfg.OUTPUT_DIR), # Output -> Final model saving path

        num_train_epochs=1         if args.debug else cfg.NUM_EPOCHS, # During real training it' ll be NUM_EPOCHS
        max_steps=2                if args.debug else -1, 

# If not in debug mode then follow the numbers in config.py
        per_device_train_batch_size=1  if args.debug else cfg.BATCH_GPU,
        per_device_eval_batch_size=1   if args.debug else cfg.BATCH_GPU,
        gradient_accumulation_steps=1  if args.debug else cfg.GRAD_ACCUM,  # effective batch = BATCH_GPU * NUM_GPUS * GRAD_ACCUM

        max_length=cfg.MAX_SEQ_LEN, # Truncate anything over maximum sequence length that 4096  
        learning_rate=cfg.LR,
        lr_scheduler_type="cosine", # To smoothly reduce the reduce the learning rate at the end of training
        warmup_ratio=0.0           if args.debug else cfg.WARMUP_RATIO,
        weight_decay=cfg.WEIGHT_DECAY, # Overfitting prevention with the help of L2 regularisation
        optim="adamw_torch_fused",   

        bf16=True,
        gradient_checkpointing=True,#recomputing activations during backward pass to save memory                          
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        eval_strategy="steps",
        eval_steps=1               if args.debug else cfg.EVAL_STEPS,
        save_strategy="steps",
        save_steps=9999            if args.debug else cfg.SAVE_STEPS,  
        save_total_limit=cfg.SAVE_TOTAL,         
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        assistant_only_loss=True,  # Dont compute loss on the prompt        
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

    print("\nStarting Stage1B training")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if not args.debug: # if not debug purpose
        save_path = cfg.OUTPUT_DIR / "final"
        print(f"\nSaving the adapter to {save_path} ...")
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print("All completed")


if __name__ == "__main__":
    main()
