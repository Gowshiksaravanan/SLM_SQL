
from pathlib import Path


ROOT = Path(__file__).parent.parent# To denote as project root is two levels upward from this file
MODEL_PATH= ROOT / "models" / "phi-4"           # base Phi-4 model's weights
SYNSQL_FILE = ROOT / "data" / "SynSQL-2.5M" / "data.json"        # full SynSQL-2.5M dataset
SYNSQL_DB_DIR = ROOT / "data" / "SynSQL-2.5M" / "databases"        # extracted SQLite files which is the mock of DB
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "stage1b"  # This is the path where LoRA adapter is saved
LORA_RANK = 64 # 64 rank to keep the adpater small in the same time covering all attention and MLP projections
LORA_ALPHA  = 128 # alpha = 2x
LORA_DROPOUT = 0.05
LORA_TARGETS = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # Baiscally covering all linear layers in Phi-4
MAX_SEQ_LEN  = 4096  # max tokens per sample [schema + question + CoT + SQL]
# effective batch size = BATCH_GPU * NUMBER OF GPUS * GRAD_ACCUMULATION = 8 * 6 * 2 = 96
BATCH_GPU = 8
GRAD_ACCUM = 2
NUM_GPUS = 6#Number of gpus used

NUM_EPOCHS = 3
LR = 2e-4  # standard LoRA learning rate for SFT. "Tried with 1e-4 and the loss was too slow.""
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
VAL_SPLIT = 0.02 
SEED= 42
EVAL_STEPS = 250 # How often we do the eval during training
SAVE_STEPS= 500 # Checkpoint saving steps
SAVE_TOTAL= 3 # keep only the 3 most recent checkpoints
REPORT_TO = "none" 
LOGGING_STEPS = 10
#Simple system prompt usied for every training example
SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a natural language question, "
    "think step-by-step inside <think>...</think> tags to identify the relevant tables "
    "and columns, reason through the query logic, then write the final SQLite SQL query.")
