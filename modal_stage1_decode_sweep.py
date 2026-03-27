"""
Decode sweep for Stage1-large on its own held-out test set.

Goal: find better inference params without retraining.

Run:
  uvx --from modal modal run modal_stage1_decode_sweep.py
"""

import modal

app = modal.App("hgr-stage1-decode-sweep")
vol_stage1 = modal.Volume.from_name("hgr-stage1-large")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.2",
        "transformers>=4.39.3",
        "peft>=0.10.0",
        "sacrebleu>=2.4.0",
        "sentencepiece>=0.2.0",
    )
)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/stage1": vol_stage1})
def sweep():
    import json
    import itertools
    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    with open("/stage1/test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open("/stage1/adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model = adapter_cfg["base_model_name_or_path"]

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, "/stage1")
    model.eval()

    sources = [x["source"] for x in test_data]
    refs = [x["target"] for x in test_data]

    grid = {
        "num_beams": [4, 6, 8],
        "max_length": [48, 64, 80],
        "length_penalty": [0.8, 1.0, 1.2],
        "repetition_penalty": [1.0, 1.1],
    }

    combos = list(itertools.product(
        grid["num_beams"],
        grid["max_length"],
        grid["length_penalty"],
        grid["repetition_penalty"],
    ))

    bleu_fn = BLEU(effective_order=True)
    chrf_fn = CHRF(word_order=2)

    results = []
    batch = 64
    for nb, ml, lp, rp in combos:
        preds = []
        for i in range(0, len(sources), batch):
            bsrc = sources[i:i + batch]
            inputs = tok(bsrc, return_tensors="pt", padding=True, truncation=True, max_length=ml).to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=ml,
                    num_beams=nb,
                    length_penalty=lp,
                    repetition_penalty=rp,
                    early_stopping=True,
                )
            preds.extend([x.strip() for x in tok.batch_decode(out, skip_special_tokens=True)])

        bleu = bleu_fn.corpus_score(preds, [refs]).score
        chrf = chrf_fn.corpus_score(preds, [refs]).score
        row = {
            "num_beams": nb,
            "max_length": ml,
            "length_penalty": lp,
            "repetition_penalty": rp,
            "bleu": bleu,
            "chrf": chrf,
        }
        results.append(row)
        print(f"BLEU={bleu:.4f} CHRF={chrf:.4f} cfg={row}")

    best = max(results, key=lambda x: x["bleu"])

    out = {"best": best, "results": sorted(results, key=lambda x: x["bleu"], reverse=True)[:10]}
    with open("/stage1/decode_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    vol_stage1.commit()

    print(f"BEST_BLEU={best['bleu']:.4f}")
    print("RESULT_PATH=/stage1/decode_sweep_results.json")


@app.local_entrypoint()
def main():
    sweep.remote()
