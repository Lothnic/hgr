"""
Evaluate Stage 2 model with the same protocol used in Stage 1 test evaluation.

Protocol matched from Stage 1:
- test set: stage1_large test_set.json
- generation: num_beams=4, max_length=48
- metrics: BLEU and chrF++ (sacrebleu)

Run:
  uvx --from modal modal run modal_evaluate_stage2_same_protocol.py
"""

import modal

app = modal.App("hgr-eval-stage2-same-protocol")

vol_stage1_large = modal.Volume.from_name("hgr-stage1-large")
vol_stage2_dpo = modal.Volume.from_name("hgr-stage2-dpo")

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


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        "/stage1_large_output": vol_stage1_large,
        "/stage2_dpo_output": vol_stage2_dpo,
    },
)
def evaluate():
    import json
    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    test_path = "/stage1_large_output/test_set.json"
    model_dir = "/stage2_dpo_output/final_model"

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(f"{model_dir}/adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()

    sources = [item["source"] for item in test_data]
    references = [item["target"] for item in test_data]
    directions = [item.get("direction", "unknown") for item in test_data]

    preds = []
    batch_size = 64
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i : i + batch_size]
        inputs = tokenizer(
            batch_src,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=48,
        ).to("cuda")

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=48,
                num_beams=4,
                early_stopping=True,
            )
        dec = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend([x.strip() for x in dec])

    bleu_fn = BLEU(effective_order=True)
    chrf_fn = CHRF(word_order=2)

    overall_bleu = bleu_fn.corpus_score(preds, [references]).score
    overall_chrf = chrf_fn.corpus_score(preds, [references]).score

    by_dir = {}
    for p, r, d in zip(preds, references, directions):
        key = d if isinstance(d, str) and d else "unknown"
        by_dir.setdefault(key, {"p": [], "r": []})
        by_dir[key]["p"].append(p)
        by_dir[key]["r"].append(r)

    per_direction = {}
    for d, data in by_dir.items():
        if not data["p"]:
            continue
        b = bleu_fn.corpus_score(data["p"], [data["r"]]).score
        c = chrf_fn.corpus_score(data["p"], [data["r"]]).score
        per_direction[d] = {
            "bleu": b,
            "chrf": c,
            "n": len(data["p"]),
        }

    results = {
        "overall_bleu": overall_bleu,
        "overall_chrf": overall_chrf,
        "per_direction": per_direction,
        "protocol": {
            "num_beams": 4,
            "max_length": 48,
            "test_set": test_path,
            "model_dir": model_dir,
        },
    }

    out_path = "/stage2_dpo_output/stage2_same_protocol_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    vol_stage2_dpo.commit()

    print(f"OVERALL_BLEU={overall_bleu:.4f}")
    print(f"OVERALL_CHRF={overall_chrf:.4f}")
    print(f"RESULT_PATH={out_path}")


@app.local_entrypoint()
def main():
    evaluate.remote()
