import torch
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

m = "google/mt5-small"
print("Loading model...")
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
mod = AutoModelForSeq2SeqLM.from_pretrained(m, quantization_config=q, device_map={"": 0})

l = LoraConfig(r=8, lora_alpha=16, target_modules=["q", "v"], task_type=TaskType.SEQ_2_SEQ_LM)
mod = get_peft_model(mod, l)

print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MiB")
