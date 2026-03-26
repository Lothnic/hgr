"""
Generate full DPO triplets from cleaned parallel data on Modal.

Run:
  uvx --from modal modal run modal_stage2_data_full_clean.py
"""

import modal

app = modal.App("hgr-stage2-data-full-clean")

vol_stage1_large = modal.Volume.from_name("hgr-stage1-large")
vol_stage2 = modal.Volume.from_name("hgr-stage2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.2",
        "transformers>=4.39.3",
        "peft>=0.10.0",
        "pandas",
        "sentencepiece>=0.2.0",
    )
    .add_local_file("src/hgr/data/parallel.filtered.csv", remote_path="/data/parallel.filtered.csv")
    .add_local_file("src/hgr/data/dataset_info.json", remote_path="/data/dataset_info.json")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/stage1_large_output": vol_stage1_large,
        "/stage2_output": vol_stage2,
    },
)
def generate_full_clean_data():
    import json
    import re
    import pandas as pd
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    with open("/data/dataset_info.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)

    src_lang = lang_data["src_lang"]
    tgt_lang = lang_data["tgt_lang"]

    df = pd.read_csv("/data/parallel.filtered.csv", encoding="utf-8")

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text.strip().strip('"').strip("'"))

    df["src"] = df["src"].apply(clean_text)
    df["tgt"] = df["tgt"].apply(clean_text)
    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)]
    df = df.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)

    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "source": f"translate {src_lang} to {tgt_lang}: " + row["src"],
            "preferred": row["tgt"],
        })
        pairs.append({
            "source": f"translate {tgt_lang} to {src_lang}: " + row["tgt"],
            "preferred": row["src"],
        })

    base_model_name = "google/mt5-large"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, "/stage1_large_output")
    model.eval()

    out = []
    batch_size = 64
    max_len = 80

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        sources = [x["source"] for x in batch]

        inputs = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to("cuda")

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=1,
                do_sample=True,
                temperature=1.2,
                top_p=0.95,
            )

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for p, row in zip(preds, batch):
            out.append({
                "source": row["source"],
                "preferred": row["preferred"],
                "unpreferred": p.strip(),
            })

    output_path = "/stage2_output/dpo_dataset_full_clean.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    meta = {
        "pairs": len(out),
        "parallel_rows_clean": len(df),
        "bidirectional": True,
        "output": output_path,
    }
    with open("/stage2_output/dpo_dataset_full_clean.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    vol_stage2.commit()
    print(f"DPO_FULL_CLEAN_PAIRS={len(out)}")


@app.local_entrypoint()
def main():
    generate_full_clean_data.remote()
