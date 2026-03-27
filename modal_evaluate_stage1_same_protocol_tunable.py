import modal

app = modal.App("hgr-eval-stage1-tunable")
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
    .add_local_file("stage1_eval_config.json", remote_path="/config/stage1_eval_config.json")
)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/stage1": vol_stage1})
def evaluate():
    import json
    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    with open("/config/stage1_eval_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with open("/stage1/test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open("/stage1/adapter_config.json", "r", encoding="utf-8") as f:
        ac = json.load(f)
    base_model_name = ac["base_model_name_or_path"]

    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, "/stage1")
    model.eval()

    sources = [x["source"] for x in test_data]
    refs = [x["target"] for x in test_data]

    batch_size = int(cfg.get("batch_size", 32))
    preds = []
    for i in range(0, len(sources), batch_size):
        bsrc = sources[i:i+batch_size]
        inputs = tok(bsrc, return_tensors="pt", padding=True, truncation=True, max_length=cfg["max_length"]).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=cfg["max_length"],
                num_beams=cfg["num_beams"],
                length_penalty=cfg["length_penalty"],
                repetition_penalty=cfg["repetition_penalty"],
                early_stopping=True,
            )
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))

    bleu = BLEU(effective_order=True).corpus_score(preds, [refs]).score
    chrf = CHRF(word_order=2).corpus_score(preds, [refs]).score

    print(f"CFG={cfg}")
    print(f"OVERALL_BLEU={bleu:.4f}")
    print(f"OVERALL_CHRF={chrf:.4f}")


@app.local_entrypoint()
def main():
    evaluate.remote()
