# This python script is to merge the adapter (fine tuned) after stage 1 with the base phi-4 model

from pathlib import Path
import torch
from peft import PeftModel                                   
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT= Path(__file__).parent.parent                              
BASE_MODEL = ROOT / "models" / "phi-4"#path of original phi-4 model                                 
ADAPTER_PATH = Path(__file__).parent / "checkpoints" / "stage1b" / "final" #Final lora adapte saved path
OUTPUT_PATH = Path(__file__).parent / "checkpoints" / "stage1b" / "merged" # Path where model need to placed after merging

# The chat template is same as training but the {% generation %} tag is a TRL loss-masking marker that does nothing at inference. This format got with help of ai
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
#Merging is done on cpu and not gpu
    base = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL),dtype=torch.bfloat16,device_map="cpu",
        trust_remote_code=True,
    )
    print("Loading Stage 1B LoRA adapter")
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))#Wrap the base model with lora layers
    # merge_and_unload folds the low-rank matrices into the base weights and removes the adapter wrapper
    print("Merging LoRA weights into model copy")
    merged = model.merge_and_unload()
    print(f"Saving merged model to {OUTPUT_PATH}")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(OUTPUT_PATH), safe_serialization=True)  # saves as safetensors shards
# Saving the tokeniser in the same path as the model output becuase it ll be easy during inference
    print("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    tokenizer.chat_template = CHAT_TEMPLATE   # overwrite with inference-safe template
    tokenizer.save_pretrained(str(OUTPUT_PATH))
    print()
    print("Done.")
if __name__ == "__main__":
    main()
