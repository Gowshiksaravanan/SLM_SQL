# SFT baseline config — schema+question -> SQL, no CoT

from pathlib import Path

ROOT        = Path(__file__).parent.parent.parent
MODEL_PATH  = ROOT / "models" / "phi-4"
DATA_DIR    = Path(__file__).parent.parent / "data"
KAGGLE_FILE = DATA_DIR / "kaggle_wide_stage1_fixed.jsonl"
WIDE_FILE   = DATA_DIR / "wide_stage1.jsonl"
OUTPUT_DIR  = Path(__file__).parent / "checkpoints"

MAX_TRAIN_SAMPLES = 10_000
VAL_SPLIT         = 0.02
SEED              = 42

LORA_RANK    = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

MAX_SEQ_LEN = 4096

BATCH_GPU  = 8
GRAD_ACCUM = 2
NUM_EPOCHS = 1
LR         = 2e-4
WARMUP_RATIO  = 0.03
WEIGHT_DECAY  = 0.01

EVAL_STEPS    = 50
SAVE_STEPS    = 100
SAVE_TOTAL    = 2

REPORT_TO     = "none"
LOGGING_STEPS = 10

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a natural language question, "
    "write the correct SQLite SQL query."
)
