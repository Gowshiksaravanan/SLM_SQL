# Two-Stage Text-to-SQL Pipeline

Train Phi-4 (14B) on Text-to-SQL through chain-of-thought supervised fine-tuning followed by Group Relative Policy Optimization (GRPO) with an execution-based reward function.

## Requirements

- Python >= 3.11
- CUDA >= 12.1
- PyTorch >= 2.4 with CUDA support
- 6× H200 GPUs for training, 1× GPU with 40GB+ VRAM for inference

Install dependencies:


pip install transformers peft trl accelerate datasets sqlglot torch ijson


## Data

Download the datasets from their original sources:

- **Phi-4 model weights** — [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) on HuggingFace
- **SynSQL-2.5M** — [SynSQL-2.5M](https://huggingface.co/datasets/Chinchilla-Research/SynSQL-2.5M) on HuggingFace. You need both `data.json` and the `databases.zip` (SQLite files).
- **Spider** — [Yale Semantic Parsing](https://yale-lily.github.io/spider). Download the data and extract to `data/spider_official/spider_data/`.

## Setup

Put the models and data files in the following directory layout:


models/phi-4/                              # Phi-4 model weights
data/SynSQL-2.5M/databases/                # SynSQL SQLite files (extracted from databases.zip)
data/spider_official/spider_data/          # Spider
  ├── database/
  ├── dev.json
  └── tables.json


File paths are hard-coded in `stage1/config.py` and `stage2/config_stage2.py`. If you keep your files in another location, you must modify those files.



## Running the pipeline

**Stage 1 — CoT SFT**


accelerate launch --config_file configs/accelerate_6gpu.yaml stage1/train_stage1b.py
python stage1/merge_stage1b.py


The merging process embeds the LoRA adapter into base Phi-4. Stage 2 requires this process to be completed first.

**Prepare Stage 2 dataset**


python data/prepare_stage2.py --target 50000


Filters SynSQL to ~50K complex queries and validates each one runs against its database. To oversample hard patterns (UNION/INTERSECT/EXCEPT/HAVING) before filtering:


python data/filter_hard_records.py

**Stage 2 — GRPO**


accelerate launch --config_file configs/accelerate_6gpu.yaml stage2/train_stage2.py
python stage2/merge_stage2.py


**Evaluation**


python inference/infer_stage2.py --checkpoint stage2/merged


Add `--n 50` for a quick sanity check on a small subset.

To evaluate a Stage 1 checkpoint mid-training (before merge):


python inference/infer_checkpoint.py --checkpoint stage1/checkpoints/stage1b/checkpoint-500 --n 100


Omit `--checkpoint` to evaluate the base Phi-4 model. Use `--indices` to run specific Spider dev examples by index.

**SFT baseline (optional, for comparison)**


accelerate launch --config_file configs/accelerate_6gpu.yaml sft_baseline/train.py
python sft_baseline/infer.py --checkpoint sft_baseline/checkpoints/final


## Reward functions

The GRPO reward is in `stage2/rewards/combined_reward.py`. Cascading three-tier structure: format gate → execution check → proxy fallback. Each component file under `stage2/rewards/` has a `__main__` block with unit tests:


python stage2/rewards/execution.py
python stage2/rewards/format_reward.py
python stage2/rewards/schema_reward.py
python stage2/rewards/structural.py

