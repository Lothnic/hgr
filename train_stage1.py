#!/usr/bin/env python3
"""
Stage 1: Supervised Fine-Tuning (SFT) of mT5 on local GPU.
LoRA adapters trained on bidirectional parallel pairs.

Usage:
    uv run python train_stage1.py                          # defaults
    uv run python train_stage1.py --epochs 5 --batch 32    # override
    uv run python train_stage1.py --model google/mt5-base   # smaller model
"""
import argparse
import json
import logging
import math
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from hgr.config import ModelConfig, TrainingConfig

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().strip("\"'"))


def build_pairs(df, src_lang, tgt_lang, src_col="src", tgt_col="tgt"):
    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "source": f"translate {src_lang} to {tgt_lang}: " + row[src_col],
            "target": row[tgt_col],
            "direction": "src2tgt",
        })
    for _, row in df.iterrows():
        pairs.append({
            "source": f"translate {tgt_lang} to {src_lang}: " + row[tgt_col],
            "target": row[src_col],
            "direction": "tgt2src",
        })
    return pairs


def split_pairs(pairs, train_ratio=0.80, val_ratio=0.10, seed=42):
    src2tgt = [p for p in pairs if p["direction"] == "src2tgt"]
    tgt2src = [p for p in pairs if p["direction"] == "tgt2src"]

    def _split(lst):
        rng = random.Random(seed)
        rng.shuffle(lst)
        n = len(lst)
        t = int(n * train_ratio)
        v = t + int(n * val_ratio)
        return lst[:t], lst[t:v], lst[v:]

    s_tr, s_v, s_te = _split(src2tgt)
    t_tr, t_v, t_te = _split(tgt2src)
    train = s_tr + t_tr
    val = s_v + t_v
    test = s_te + t_te
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    return train, val, test


def evaluate(model, tokenizer, test_data, batch_size, max_length):
    bleu_fn = BLEU(effective_order=True)
    chrf_fn = CHRF(word_order=2)

    device = next(model.parameters()).device
    model.eval()
    all_p, all_r = [], []
    by_dir = {"src2tgt": {"p": [], "r": []}, "tgt2src": {"p": [], "r": []}}

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i : i + batch_size]
        sources = [b["source"] for b in batch]
        targets = [b["target"] for b in batch]
        dirs = [b["direction"] for b in batch]

        inputs = tokenizer(
            sources, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_length=max_length,
                num_beams=4, early_stopping=True,
            )
        preds = [p.strip() for p in tokenizer.batch_decode(out, skip_special_tokens=True)]
        for p, r, d in zip(preds, targets, dirs):
            all_p.append(p)
            all_r.append(r)
            by_dir[d]["p"].append(p)
            by_dir[d]["r"].append(r)

    overall_bleu = bleu_fn.corpus_score(all_p, [all_r]).score
    overall_chrf = chrf_fn.corpus_score(all_p, [all_r]).score
    logger.info(f"Overall  BLEU: {overall_bleu:.4f}  chrF++: {overall_chrf:.4f}")

    results = {"overall_bleu": overall_bleu, "overall_chrf": overall_chrf, "per_direction": {}}
    for d, data in by_dir.items():
        if not data["p"]:
            continue
        b = bleu_fn.corpus_score(data["p"], [data["r"]]).score
        c = chrf_fn.corpus_score(data["p"], [data["r"]]).score
        logger.info(f"  {d}  BLEU: {b:.4f}  chrF++: {c:.4f}")
        results["per_direction"][d] = {"bleu": b, "chrf": c, "n": len(data["p"])}

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 1 — SFT with LoRA (local GPU)")
    parser.add_argument("--data", default="src/hgr/data/parallel.csv")
    parser.add_argument("--lang-info", default="src/hgr/data/dataset_info.json")
    parser.add_argument("--model", default="google/mt5-large")
    parser.add_argument("--output", default="stage1_output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate existing adapter")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.lang_info, "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    src_lang, tgt_lang = lang_data["src_lang"], lang_data["tgt_lang"]

    logger.info(f"Loading {src_lang} <-> {tgt_lang} from {args.data}")
    df = pd.read_csv(args.data, encoding="utf-8")
    df["src"] = df["src"].apply(clean_text)
    df["tgt"] = df["tgt"].apply(clean_text)
    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)]
    df = df.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)
    logger.info(f"Clean pairs: {len(df)}")

    pairs = build_pairs(df, src_lang, tgt_lang)
    logger.info(f"Bidirectional pairs: {len(pairs)}")
    train_data, val_data, test_data = split_pairs(pairs, seed=args.seed)
    logger.info(f"Train {len(train_data)} | Val {len(val_data)} | Test {len(test_data)}")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "test_set.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    dataset = DatasetDict({
        "train": HFDataset.from_list(train_data),
        "validation": HFDataset.from_list(val_data),
        "test": HFDataset.from_list(test_data),
    })

    logger.info(f"Loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        lora_dropout=0.1, bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(examples):
        inputs = tokenizer(
            examples["source"], max_length=args.max_len,
            truncation=True, padding=False,
        )
        labels = tokenizer(
            examples["target"], max_length=args.max_len,
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

    total_steps = math.ceil(len(tokenized["train"]) / args.batch) * args.epochs
    warmup_steps = int(total_steps * 0.10)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        optim="adamw_torch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        save_total_limit=2,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    logger.info(f"Training | batch={args.batch} | lr={args.lr}")
    if not args.eval_only:
        trainer.train()

    if not args.eval_only:
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        logger.info(f"Model saved to {args.output}")
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            AutoModelForSeq2SeqLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16,
                device_map={"": 0} if torch.cuda.is_available() else None,
            ),
            args.output,
        )
        logger.info(f"Loaded adapter from {args.output} for eval")

    results = evaluate(model, tokenizer, test_data, args.eval_batch, args.max_len)
    results["config"] = {
        "model": args.model, "epochs": args.epochs,
        "batch": args.batch, "lr": args.lr,
        "max_len": args.max_len, "seed": args.seed,
    }
    with open(os.path.join(args.output, "stage1_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    samples = random.sample(test_data, min(20, len(test_data)))
    sample_out = []
    for s in samples:
        inp = tokenizer(s["source"], return_tensors="pt", truncation=True, max_length=args.max_len)
        if torch.cuda.is_available():
            inp = {k: v.to("cuda") for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_length=args.max_len, num_beams=4, early_stopping=True)
        sample_out.append({
            "source": s["source"], "reference": s["target"],
            "prediction": tokenizer.decode(out[0], skip_special_tokens=True).strip(),
            "direction": s["direction"],
        })
    with open(os.path.join(args.output, "stage1_samples.json"), "w", encoding="utf-8") as f:
        json.dump(sample_out, f, ensure_ascii=False, indent=2)

    logger.info("Stage 1 complete.")


if __name__ == "__main__":
    main()
