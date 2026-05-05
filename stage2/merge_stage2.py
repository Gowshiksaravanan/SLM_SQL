# This is for getting the final model by merging stage 1b with the fine tuned model

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parent_dir    = Path(__file__).parent.parent
BASE_MODEL   = parent_dir / "stage1" / "checkpoints" / "stage1b" / "merged"
ADAPTER_PATH = Path(__file__).parent / "checkpoints" / "stage2" / "final"
OUTPUT_PATH  = Path(__file__).parent / "checkpoints" / "stage2" / "merged"

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


def main():
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        dtype=torch.bfloat16,
        device_map="cpu", # Merge doesnt need gpu
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))# Wrap base model with the Stage 2 LoRA adapter
    merged = model.merge_and_unload()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(OUTPUT_PATH), safe_serialization=True)  # saves as safetensors
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))  # Save tokenizer alongside the model so inference only needs one path and will be easy
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(str(OUTPUT_PATH))
    print("Merge done")


if __name__ == "__main__":
    main()
