"""
Stage 1: Supervised Fine-Tuning (SFT) of mt5-large on Modal
Multi-Language Translation with LoRA (PEFT)
=================================================
Run:   modal run --detach modal_stage1_large.py
Fetch: modal volume get hgr-stage1-large / ./stage1_large_output/
"""

import modal

# ── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("hgr-stage1-clean-training")

vol = modal.Volume.from_name("hgr-stage1-clean", create_if_missing=True)

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
    .add_local_file("src/hgr/data/parallel.filtered.csv", remote_path="/data/parallel.csv")
    .add_local_file("src/hgr/data/dataset_info.json", remote_path="/data/dataset_info.json")
)

# ── Training function ─────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="H100",
    timeout=21600,  # 6 hour safety limit for H100
    volumes={"/output": vol},
)
def train():
    import os, re, json, random, logging, math
    import numpy as np
    import pandas as pd
    from dataclasses import dataclass

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
    )
    from peft import get_peft_model, LoraConfig, TaskType
    from datasets import Dataset as HFDataset, DatasetDict
    from sacrebleu.metrics import BLEU, CHRF

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # ── Config ────────────────────────────────────────────────────────────────
    @dataclass
    class Config:
        data_path:    str = "/data/parallel.csv"
        lang_info:    str = "/data/dataset_info.json"
        output_dir:   str = "/output"
        source_col:   str = "src"
        target_col:   str = "tgt"
        model_name:   str = "google/mt5-large"

        epochs:                     int   = 5
        train_batch_size:           int   = 256 # Aggressive scale up for 80GB H100
        eval_batch_size:            int   = 256
        gradient_accumulation_steps:int   = 1     # effective batch = 128
        learning_rate:              float = 1e-4
        warmup_ratio:               float = 0.15  # more warmup for stability
        weight_decay:               float = 0.01
        max_input_length:           int   = 48
        max_target_length:          int   = 48

        train_ratio:      float = 0.80
        val_ratio:        float = 0.10
        test_ratio:       float = 0.10
        both_directions:  bool  = True

        eval_steps:                 int  = 500
        save_steps:                 int  = 500
        logging_steps:              int  = 50
        load_best_model_at_end:     bool = True
        metric_for_best_model:      str  = "eval_loss"
        greater_is_better:          bool = False
        early_stopping_patience:    int  = 5  # more patience to let model learn

        seed: int = 42

    cfg = Config()

    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text.strip().strip("\"'"))

    # ── Data ──────────────────────────────────────────────────────────────────
    with open(cfg.lang_info, "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    src_lang = lang_data["src_lang"]
    tgt_lang = lang_data["tgt_lang"]

    logger.info(f"Loading {src_lang} <-> {tgt_lang} mappings from: {cfg.data_path}")
    df = pd.read_csv(cfg.data_path, encoding="utf-8")
    for col in [cfg.source_col, cfg.target_col]:
        assert col in df.columns, f"Column '{col}' missing. Found: {list(df.columns)}"
    df[cfg.source_col] = df[cfg.source_col].apply(clean_text)
    df[cfg.target_col]    = df[cfg.target_col].apply(clean_text)
    df = df[
        (df[cfg.source_col].str.len() > 0) &
        (df[cfg.target_col].str.len() > 0)
    ].drop_duplicates(subset=[cfg.source_col, cfg.target_col]).reset_index(drop=True)
    logger.info(f"Clean pairs: {len(df)}")

    # Build pairs
    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "source": f"translate {src_lang} to {tgt_lang}: " + row[cfg.source_col],
            "target": row[cfg.target_col],
            "direction": "src2tgt",
        })
    if cfg.both_directions:
        for _, row in df.iterrows():
            pairs.append({
                "source": f"translate {tgt_lang} to {src_lang}: " + row[cfg.target_col],
                "target": row[cfg.source_col],
                "direction": "tgt2src",
            })
    logger.info(f"Total pairs: {len(pairs)}")

    # Split
    src2tgt = [p for p in pairs if p["direction"] == "src2tgt"]
    tgt2src = [p for p in pairs if p["direction"] == "tgt2src"]

    def _split(lst):
        random.shuffle(lst)
        n = len(lst)
        t = int(n * cfg.train_ratio)
        v = t + int(n * cfg.val_ratio)
        return lst[:t], lst[t:v], lst[v:]

    src2tgt_tr, src2tgt_v, src2tgt_te = _split(src2tgt)
    tgt2src_tr, tgt2src_v, tgt2src_te = _split(tgt2src)
    train_data = src2tgt_tr + tgt2src_tr
    val_data   = src2tgt_v  + tgt2src_v
    test_data  = src2tgt_te + tgt2src_te
    random.shuffle(train_data)
    random.shuffle(val_data)
    logger.info(f"Train {len(train_data)} | Val {len(val_data)} | Test {len(test_data)}")

    # Save test set
    os.makedirs(cfg.output_dir, exist_ok=True)
    test_path = os.path.join(cfg.output_dir, "test_set.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Test set saved: {test_path}")

    splits = {"train": train_data, "validation": val_data, "test": test_data}
    dataset = DatasetDict({k: HFDataset.from_list(v) for k, v in splits.items()})
    logger.info(f"\n{dataset}")

    # ── Model & Tokenizer ─────────────────────────────────────────────────────
    logger.info(f"Loading {cfg.model_name} in bfloat16 + LoRA ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # bf16 — 1.2B Parameters fits onto 24GB VRAM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Tokenize ──────────────────────────────────────────────────────────────
    def tokenize(examples):
        inputs = tokenizer(
            examples["source"],
            max_length=cfg.max_input_length,
            truncation=True, padding=False,
        )
        labels = tokenizer(
            examples["target"],
            max_length=cfg.max_target_length,
            truncation=True, padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(
        tokenize, batched=True,
        remove_columns=["source", "target", "direction"],
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        label_pad_token_id=-100, pad_to_multiple_of=8,
    )

    # ── Metrics (used only for final test evaluation) ─────────────────────────
    bleu_fn = BLEU(effective_order=True)
    chrf_fn = CHRF(word_order=2)

    # ── Training ──────────────────────────────────────────────────────────────
    total_steps = math.ceil(len(tokenized["train"]) / cfg.train_batch_size) // cfg.gradient_accumulation_steps * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,

        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        optim="adamw_torch",

        bf16=True,  # bf16 prevents loss underflow that causes train_loss=0.0
        gradient_checkpointing=True,

        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=2,

        predict_with_generate=False,  # use eval_loss, not generation

        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,

        seed=cfg.seed,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    logger.info(
        f"Training | effective batch = "
        f"{cfg.train_batch_size * cfg.gradient_accumulation_steps}"
    )
    result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    logger.info(f"Model saved: {cfg.output_dir}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 BASELINE — TEST SET RESULTS")
    logger.info("=" * 60)

    model.eval()
    device = next(model.parameters()).device
    all_p, all_r = [], []
    by_dir = {"src2tgt": {"p": [], "r": []}, "tgt2src": {"p": [], "r": []}}

    for i in range(0, len(test_data), cfg.eval_batch_size):
        batch   = test_data[i : i + cfg.eval_batch_size]
        sources = [b["source"]    for b in batch]
        targets = [b["target"]    for b in batch]
        dirs    = [b["direction"] for b in batch]

        inputs = tokenizer(
            sources, return_tensors="pt",
            max_length=cfg.max_input_length,
            truncation=True, padding=True,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=cfg.max_target_length,
                num_beams=4, early_stopping=True,
            )

        preds = [p.strip() for p in tokenizer.batch_decode(out, skip_special_tokens=True)]
        for p, r, d in zip(preds, targets, dirs):
            all_p.append(p); all_r.append(r)
            by_dir[d]["p"].append(p); by_dir[d]["r"].append(r)

    bleu = bleu_fn.corpus_score(all_p, [all_r])
    chrf = chrf_fn.corpus_score(all_p, [all_r])
    logger.info(f"Overall  BLEU: {bleu.score:.4f}  chrF++: {chrf.score:.4f}")

    results = {"overall_bleu": bleu.score, "overall_chrf": chrf.score, "per_direction": {}}
    labels  = {"src2tgt": f"{src_lang}->{tgt_lang}", "tgt2src": f"{tgt_lang}->{src_lang}"}

    for d, data in by_dir.items():
        if not data["p"]:
            continue
        b = bleu_fn.corpus_score(data["p"], [data["r"]])
        c = chrf_fn.corpus_score(data["p"], [data["r"]])
        logger.info(f"{labels[d]}  BLEU: {b.score:.4f}  chrF++: {c.score:.4f}")
        results["per_direction"][d] = {"bleu": b.score, "chrf": c.score, "n": len(data["p"])}

    with open(os.path.join(cfg.output_dir, "stage1_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Sample translations
    samples = random.sample(list(zip(all_p, all_r, test_data)), min(30, len(all_p)))
    sample_out = [
        {"source": item["source"], "reference": r, "prediction": p, "direction": item["direction"]}
        for p, r, item in samples
    ]
    with open(os.path.join(cfg.output_dir, "stage1_samples.json"), "w", encoding="utf-8") as f:
        json.dump(sample_out, f, ensure_ascii=False, indent=2)

    # Save config
    with open(os.path.join(cfg.output_dir, "stage1_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Commit volume so outputs persist
    vol.commit()

    logger.info("=" * 60)
    logger.info("Stage 1 COMPLETE")
    logger.info(f"  Best model  : {cfg.output_dir}")
    logger.info(f"  Test set    : {cfg.output_dir}/test_set.json")
    logger.info("  Next step   : stage2_generate_unpreferred.py")
    logger.info("=" * 60)
    logger.info("Results saved to Modal Volume 'hgr-stage1-large'")
    logger.info("Download with: modal volume get hgr-stage1-large / ./stage1_large_output/")


# ── Entrypoint ────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    train.remote()
