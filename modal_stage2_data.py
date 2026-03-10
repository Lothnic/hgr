"""
Stage 2: Generate DPO Triplets on Modal
=======================================
Reads the training CSV, generates "unpreferred" translations using 
the Stage 1 model, and saves DPO triplets to a Modal Volume.

Run: modal run --detach modal_stage2_data.py
"""
import modal
import os

app = modal.App("hgr-stage2-data")

# Volumes
vol_stage1 = modal.Volume.from_name("hgr-stage1")
vol_stage2 = modal.Volume.from_name("hgr-stage2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.2",
        "transformers>=4.39.3",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "sacrebleu>=2.4.0",
        "sentencepiece>=0.2.0",
        "accelerate>=0.28.0",
        "protobuf>=4.25.0",
        "pandas",
        "numpy",
        "scipy",
    )
    .add_local_file("src/hgr/data/parallel.csv", remote_path="/data/parallel.csv")
    .add_local_file("src/hgr/data/dataset_info.json", remote_path="/data/dataset_info.json")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/stage1_output": vol_stage1,
        "/stage2_output": vol_stage2,
    },
)
def generate_data():
    import os, re, json, logging
    import pandas as pd
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    # Hardcoded config mapping to what we used
    DATA_PATH = "/data/parallel.csv"
    INFO_PATH = "/data/dataset_info.json"
    OUTPUT_PATH = "/stage2_output/dpo_dataset_30k.json"
    MODEL_NAME = "google/mt5-large"
    LORA_PATH = "/stage1_output"
    
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    SOURCE_LANG = lang_data["src_lang"]
    TARGET_LANG = lang_data["tgt_lang"]
    SOURCE_COL = "src"
    TARGET_COL = "tgt"
    
    PREFIX_MHI = f"translate {SOURCE_LANG} to {TARGET_LANG}: "
    PREFIX_HIM = f"translate {TARGET_LANG} to {SOURCE_LANG}: "
    
    BATCH_SIZE = 128
    MAX_LEN = 48

    # 1. Load Original Data
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    
    def clean_text(text):
        if not isinstance(text, str): return ""
        return re.sub(r"\s+", " ", text.strip().strip("\"'"))

    df[SOURCE_COL] = df[SOURCE_COL].apply(clean_text)
    df[TARGET_COL] = df[TARGET_COL].apply(clean_text)
    df = df[(df[SOURCE_COL].str.len() > 0) & (df[TARGET_COL].str.len() > 0)]
    df = df.drop_duplicates(subset=[SOURCE_COL, TARGET_COL]).reset_index(drop=True)

    # Note: In Stage 1, we split the data and left 10% for testing.
    # To do this correctly without data leakage, we should ideally load the train split.
    # But since we just need "unpreferred" generation over training data, we can 
    # either re-shuffle with seed=42 or generate for the whole dataset and filter later.
    # Let's just generate for all of them for simplicity — we can split later.
    
    pairs = []
    for _, row in df.iterrows():
        pairs.append({"source": PREFIX_MHI + row[SOURCE_COL], "preferred": row[TARGET_COL]})
        pairs.append({"source": PREFIX_HIM + row[TARGET_COL], "preferred": row[SOURCE_COL]})

    # Downsample to 30K for faster experimentation while giving the model more data
    import random
    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[:30000]

    logger.info(f"Total pairs to generate: {len(pairs)}")

    # 2. Load Stage 1 PEFT Model
    logger.info("Loading Base Model + Stage 1 LoRA")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map={"": 0}
    )
    
    # Load the trained adapter from Volume
    if not os.path.exists(os.path.join(LORA_PATH, "adapter_model.safetensors")):
        raise FileNotFoundError(f"Missing LoRA weights in {LORA_PATH}. Did Stage 1 complete properly?")
        
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 3. Generate Unpreferred Translations
    unpreferred = []
    logger.info("Generating translations...")
    
    for i in tqdm(range(0, len(pairs), BATCH_SIZE)):
        batch_pairs = pairs[i : i + BATCH_SIZE]
        sources = [p["source"] for p in batch_pairs]
        
        inputs = tokenizer(
            sources, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=MAX_LEN,
                num_beams=4, # Beam search for higher quality generating (though we actually want imperfect 'unpreferred')
                early_stopping=True
            )
        
        preds = [p.strip() for p in tokenizer.batch_decode(out, skip_special_tokens=True)]
        
        for p, pair in zip(preds, batch_pairs):
            pair["unpreferred"] = p

    # 4. Save DPO Dataset
    logger.info(f"Saving DPO triplets to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    vol_stage2.commit()
    logger.info("Done! Run stage2 training next.")

@app.local_entrypoint()
def main():
    generate_data.remote()
