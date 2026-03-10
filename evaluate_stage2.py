import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from tqdm import tqdm
from src.hgr.evaluation.metrics import evaluate_all

def main():
    print("Loading test data...")
    with open("stage1_output/test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open("src/hgr/data/dataset_info.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    src_lang = lang_data["src_lang"]
    tgt_lang = lang_data["tgt_lang"]

    # Base model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "google/mt5-large"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16, 
        device_map={"": 0}
    )

    # Load Stage 2 Model (Adapter on top of Base)
    print("Loading Stage 2 Weights from ./stage2_output/ ...")
    model = PeftModel.from_pretrained(base_model, "stage2_output/")
    model.eval()

    sources = [item["source"] for item in test_data]
    references = [item["target"] for item in test_data]
    predictions = []

    print(f"Translating {len(test_data)} test sentences with Stage 2 model...")
    batch_size = 64
    for i in tqdm(range(0, len(sources), batch_size)):
        batch_src = sources[i:i+batch_size]
        inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=48).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=48, 
                num_beams=1,  # Keep it fast for eval
                early_stopping=True
            )
        
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(preds)

    # Separate by direction for metrics
    src2tgt_preds, src2tgt_refs = [], []
    tgt2src_preds, tgt2src_refs = [], []

    for src, pred, ref in zip(sources, predictions, references):
        if f"{src_lang} to {tgt_lang}" in src:
            src2tgt_preds.append(pred)
            src2tgt_refs.append(ref)
        elif f"{tgt_lang} to {src_lang}" in src:
            tgt2src_preds.append(pred)
            tgt2src_refs.append(ref)

    print("\n=== FINAL STAGE 2 METRICS ===")
    
    print(f"\n--- {src_lang} → {tgt_lang} ---")
    if src2tgt_preds:
        src2tgt_metrics = evaluate_all(src2tgt_preds, src2tgt_refs, metrics=["bleu", "chrf"])
        print(f"BLEU: {src2tgt_metrics['bleu']:.2f}")
        print(f"chrF++: {src2tgt_metrics['chrf']:.2f}")
    else:
        print("No samples found.")

    print(f"\n--- {tgt_lang} → {src_lang} ---")
    if tgt2src_preds:
        tgt2src_metrics = evaluate_all(tgt2src_preds, tgt2src_refs, metrics=["bleu", "chrf"])
        print(f"BLEU: {tgt2src_metrics['bleu']:.2f}")
        print(f"chrF++: {tgt2src_metrics['chrf']:.2f}")
    else:
        print("No samples found.")

    print("\n--- OVERALL ---")
    overall_metrics = evaluate_all(predictions, references, metrics=["bleu", "chrf"])
    print(f"BLEU: {overall_metrics['bleu']:.2f}")
    print(f"chrF++: {overall_metrics['chrf']:.2f}")

if __name__ == "__main__":
    main()
