import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import sys

print("Loading tokenizer...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
print("Loading base model...", file=sys.stderr)
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", torch_dtype=torch.float32)
print("Loading PEFT model...", file=sys.stderr)
model = PeftModel.from_pretrained(base_model, "./stage2_output/")

print("Checking weights for NaNs...", file=sys.stderr)
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN found in {name}")
        sys.exit(0)

print("No NaNs found in model weights.", file=sys.stderr)
sys.exit(0)
