from pathlib import Path

ROOT= Path(__file__).parent.parent.parent   # phi4-test project root
parent_dir = Path(__file__).parent.parent          # phi4-test/paper

MODEL_PATH= ROOT / "models" / "phi-4"
STAGE1B_MERGED= parent_dir / "stage1" / "checkpoints" / "stage1b" / "merged"  # output of merge_stage1b.py

OUTPUT_DIR= Path(__file__).parent / "checkpoints" / "stage2"               # stage2 LoRA checkpoints
STAGE2_MERGED = Path(__file__).parent / "checkpoints" / "stage2" / "merged"   # output of merge_stage2.py

STAGE2_DATA= parent_dir / "data" / "stage2_grpo.jsonl"  # built by data/prepare_stage2.py

SYNSQL_DB_DIR= ROOT / "data" / "SynSQL-2.5M" / "databases"
SPIDER_DB_DIR= ROOT / "data" / "spider_official" / "spider_data" / "database"
EMPTY_DB_FILE = ROOT / "pre-processing" / "database_analysis" / "data" / "empty_databases.json"

NUM_GENERATIONS = 8
BETA = 0.04
MAX_PROMPT_LENGTH = 3072
MAX_COMPLETION_LENGTH = 4096   # <think> reasoning + SQL
BATCH_GPU = 1
GRAD_ACCUM = 8
NUM_GPUS = 7
NUM_EPOCHS = 1
LR= 5e-6
WARMUP_RATIO = 0.01
SEED= 42

EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL = 3

REPORT_TO = "none"
LOGGING_STEPS = 10

SQL_TIMEOUT_SECONDS = 5

EXECUTION_WEIGHT = 1.5
FORMAT_WEIGHT = 0.7
HALLUCINATION_WEIGHT = 0.8
COT_CONSISTENCY_WEIGHT = 0.6
CLAUSE_COVERAGE_WEIGHT = 0.5
REASONING_WEIGHT= 0.4
COMPLEXITY_WEIGHT = 0.3

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a natural language question, "
    "think step-by-step inside <think>...</think> tags to identify the relevant tables "
    "and columns, reason through the query logic, then write the final SQLite SQL query."
)
