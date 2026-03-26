"""
Cost-aware Modal BLEU benchmark for autoresearch.
Runs Stage-1 adapter inference on a deterministic sample and prints FINAL_BLEU.

Run:
  uvx --from modal modal run modal_benchmark_bleu.py
"""

import modal

app = modal.App("hgr-autoresearch-bleu")

vol_stage1 = modal.Volume.from_name("hgr-stage1")
vol_stage1_large = modal.Volume.from_name("hgr-stage1-large")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.2",
        "transformers>=4.39.3",
        "peft>=0.10.0",
        "sacrebleu>=2.4.0",
        "sentencepiece>=0.2.0",
        "pandas",
    )
    .add_local_file("src/hgr/data/parallel.csv", remote_path="/data/parallel.csv")
    .add_local_file("src/hgr/data/dataset_info.json", remote_path="/data/dataset_info.json")
    .add_local_file("autoresearch_config.json", remote_path="/data/autoresearch_config.json")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        "/stage1_output": vol_stage1,
        "/stage1_large_output": vol_stage1_large,
    },
)
def run_bleu_benchmark():
    import json
    import random
    import pandas as pd
    import torch
    from sacrebleu import corpus_bleu
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    with open("/data/autoresearch_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_volume = cfg.get("model_volume", "hgr-stage1")
    model_dir = "/stage1_output" if model_volume == "hgr-stage1" else "/stage1_large_output"

    random.seed(cfg.get("seed", 42))

    with open("/data/dataset_info.json", "r", encoding="utf-8") as f:
        lang = json.load(f)

    src_lang = lang["src_lang"]
    tgt_lang = lang["tgt_lang"]

    # Robust CSV parse: tolerate malformed lines in noisy low-resource datasets.
    df = pd.read_csv(
        "/data/parallel.csv",
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["src", "tgt"]).drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)

    pairs = []
    for _, row in df.iterrows():
        pairs.append((f"translate {src_lang} to {tgt_lang}: {row['src']}", row["tgt"]))
        pairs.append((f"translate {tgt_lang} to {src_lang}: {row['tgt']}", row["src"]))

    random.shuffle(pairs)
    sample_size = min(int(cfg.get("sample_size", 256)), len(pairs))
    pairs = pairs[:sample_size]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter_cfg_path = f"{model_dir}/adapter_config.json"
    base_model_name = "google/mt5-large"
    try:
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get("base_model_name_or_path", base_model_name)
    except FileNotFoundError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    batch_size = int(cfg.get("batch_size", 64))
    max_len = int(cfg.get("max_len", 48))

    preds, refs = [], []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        sources = [x[0] for x in batch]
        references = [x[1] for x in batch]

        inputs = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=int(cfg.get("num_beams", 1)),
                do_sample=bool(cfg.get("do_sample", False)),
                temperature=float(cfg.get("temperature", 1.0)),
                top_p=float(cfg.get("top_p", 1.0)),
                length_penalty=float(cfg.get("length_penalty", 1.0)),
                repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(cfg.get("no_repeat_ngram_size", 0)),
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend([p.strip() for p in decoded])
        refs.extend(references)

    def _normalize(s: str) -> str:
        import re
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" .", ".").replace(" ,", ",")
        return s

    if bool(cfg.get("normalize_for_bleu", False)):
        preds = [_normalize(x) for x in preds]
        refs = [_normalize(x) for x in refs]

    bleu = corpus_bleu(preds, [refs]).score
    print(f"MODEL_VOLUME={model_volume}")
    print(f"SAMPLE_SIZE={sample_size}")
    print(f"FINAL_BLEU={bleu:.4f}")


@app.local_entrypoint()
def main():
    run_bleu_benchmark.remote()
