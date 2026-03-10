import os, json, re, logging, random
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "src/hgr/data/parallel.csv"
INFO_PATH = "src/hgr/data/dataset_info.json"
OUTPUT_PATH = "stage2_output/dpo_dataset_30k_sampled.json"
MODEL_NAME = "google/mt5-large"
LORA_PATH = "stage1_output"

with open(INFO_PATH, "r", encoding="utf-8") as f:
    lang_data = json.load(f)

SOURCE_LANG = lang_data["src_lang"]
TARGET_LANG = lang_data["tgt_lang"]
SOURCE_COL = "src"
TARGET_COL = "tgt"

PREFIX_MHI = f"translate {SOURCE_LANG} to {TARGET_LANG}: "
PREFIX_HIM = f"translate {TARGET_LANG} to {SOURCE_LANG}: "

BATCH_SIZE = 64
MAX_LEN = 48
NUM_SAMPLES = 30000

def main():
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    
    def clean_text(text):
        if not isinstance(text, str): return ""
        return re.sub(r"\s+", " ", text.strip().strip("\"'"))

    df[SOURCE_COL] = df[SOURCE_COL].apply(clean_text)
    df[TARGET_COL] = df[TARGET_COL].apply(clean_text)
    df = df[(df[SOURCE_COL].str.len() > 0) & (df[TARGET_COL].str.len() > 0)]
    df = df.drop_duplicates(subset=[SOURCE_COL, TARGET_COL]).reset_index(drop=True)

    pairs = []
    for _, row in df.iterrows():
        pairs.append({"source": PREFIX_MHI + row[SOURCE_COL], "preferred": row[TARGET_COL]})
        pairs.append({"source": PREFIX_HIM + row[TARGET_COL], "preferred": row[SOURCE_COL]})

    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[:NUM_SAMPLES]

    logger.info(f"Total pairs to generate: {len(pairs)}")

    logger.info("Loading Base Model + Stage 1 LoRA")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # We use float32 or bfloat16 depending on RTX 4060 compatibility
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=dtype, 
        device_map={"": 0} if torch.cuda.is_available() else None
    )
    
    if not os.path.exists(os.path.join(LORA_PATH, "adapter_model.safetensors")):
        raise FileNotFoundError(f"Missing LoRA weights in {LORA_PATH}")
        
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    logger.info("Generating HIGH-TEMPERATURE translations for UNPREFERRED DPO target...")
    
    for i in tqdm(range(0, len(pairs), BATCH_SIZE)):
        batch_pairs = pairs[i : i + BATCH_SIZE]
        sources = [p["source"] for p in batch_pairs]
        
        # Disable padding to right
        inputs = tokenizer(
            sources, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=MAX_LEN,
                do_sample=True,          # Force sampling instead of beam search/greedy
                temperature=1.2,         # High temperature to introduce grammatical flaws 
                top_p=0.9,               # Nucleus sampling
                early_stopping=True
            )
        
        preds = [p.strip() for p in tokenizer.batch_decode(out, skip_special_tokens=True)]
        
        for p, pair in zip(preds, batch_pairs):
            pair["unpreferred"] = p

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    logger.info(f"Saving DPO triplets to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    logger.info("Done! Remember to upload this to the Modal volume before training.")

if __name__ == "__main__":
    main()
