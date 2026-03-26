"""
Stage 2 (ablation): DPO-only training on Modal (manual loop, no HGR).

Run:
  uvx --from modal modal run modal_stage2_dpo_train.py
"""

import modal

app = modal.App("hgr-stage2-dpo-only")

vol_stage1_large = modal.Volume.from_name("hgr-stage1-large")
vol_stage2_input = modal.Volume.from_name("hgr-stage2")
vol_stage2_dpo = modal.Volume.from_name("hgr-stage2-dpo", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.2",
        "transformers>=4.39.3",
        "peft>=0.10.0",
        "datasets>=2.18.0",
        "sentencepiece>=0.2.0",
    )
    .add_local_file("dpo_only_config.json", remote_path="/config/dpo_only_config.json")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/stage1_large_output": vol_stage1_large,
        "/stage2_input": vol_stage2_input,
        "/stage2_dpo_output": vol_stage2_dpo,
    },
)
def train_dpo_only():
    import json
    import os
    import random

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel

    with open("/config/dpo_only_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    data_path = cfg.get("data_path", "/stage2_input/dpo_dataset_30k_sampled.json")
    with open(data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    random.shuffle(pairs)
    n_train_cfg = int(cfg.get("n_train", 4096))
    n_train = len(pairs) if n_train_cfg <= 0 else min(n_train_cfg, len(pairs))
    pairs = pairs[:n_train]

    adapter_cfg_path = "/stage1_large_output/adapter_config.json"
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_name = adapter_cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def seq_log_probs(model, input_ids, attention_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        label_mask = labels != -100
        safe_labels = labels.clamp(min=0).unsqueeze(-1)
        token_log_probs = log_probs.gather(dim=-1, index=safe_labels).squeeze(-1)
        token_log_probs = token_log_probs * label_mask.float()
        seq_lp = token_log_probs.sum(dim=-1)
        lengths = label_mask.sum(dim=-1).float().clamp(min=1.0)
        return seq_lp / lengths

    def dpo_loss(model, ref_model, sources, preferred, unpreferred, beta, max_length):
        src_inputs = tokenizer(
            sources, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to("cuda")

        pref_labels = tokenizer(
            preferred, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).input_ids.to("cuda")
        unpref_labels = tokenizer(
            unpreferred, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).input_ids.to("cuda")

        pref_labels[pref_labels == tokenizer.pad_token_id] = -100
        unpref_labels[unpref_labels == tokenizer.pad_token_id] = -100

        pol_pref = seq_log_probs(model, src_inputs.input_ids, src_inputs.attention_mask, pref_labels)
        pol_unpref = seq_log_probs(model, src_inputs.input_ids, src_inputs.attention_mask, unpref_labels)

        with torch.no_grad():
            ref_pref = seq_log_probs(ref_model, src_inputs.input_ids, src_inputs.attention_mask, pref_labels)
            ref_unpref = seq_log_probs(ref_model, src_inputs.input_ids, src_inputs.attention_mask, unpref_labels)

        pref_ratio = pol_pref - ref_pref
        unpref_ratio = pol_unpref - ref_unpref
        return -F.logsigmoid(beta * (pref_ratio - unpref_ratio)).mean()

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, "/stage1_large_output", is_trainable=True)

    ref_base = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    ref_model = PeftModel.from_pretrained(ref_base, "/stage1_large_output")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 5e-6)),
    )

    batch_size = int(cfg.get("batch_size", 2))
    grad_accum = int(cfg.get("grad_accum", 8))
    max_steps = int(cfg.get("max_steps", 60))
    max_len = int(cfg.get("max_source_length", 64))
    beta = float(cfg.get("dpo_beta", 0.1))

    model.train()
    loader = DataLoader(pairs, batch_size=batch_size, shuffle=True)

    step = 0
    optimizer.zero_grad()
    while step < max_steps:
        for batch in loader:
            sources = batch["source"]
            preferred = batch["preferred"]
            unpreferred = batch["unpreferred"]

            loss = dpo_loss(model, ref_model, sources, preferred, unpreferred, beta=beta, max_length=max_len)
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                print(f"STEP={step} LOSS={loss.item():.4f}")
            if step >= max_steps:
                break

    out_dir = "/stage2_dpo_output/final_model"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(f"{out_dir}/dpo_train_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_train": n_train,
                "learning_rate": cfg.get("learning_rate", 5e-6),
                "dpo_beta": cfg.get("dpo_beta", 0.1),
                "max_steps": max_steps,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
            },
            f,
            indent=2,
        )

    vol_stage2_dpo.commit()
    print("DPO_ONLY_TRAIN_DONE=1")


@app.local_entrypoint()
def main():
    train_dpo_only.remote()
