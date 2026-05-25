#!/usr/bin/env python3
"""
Stage 2a: Generate DPO triplets locally using the Stage 1 adapter.

Generates "unpreferred" translations via high-temperature sampling from the
Stage 1 model, producing (source, preferred, unpreferred) triplets for DPO.

Usage:
    uv run python local_stage2_data.py
    uv run python local_stage2_data.py --lora stage1_output --output dpo_pairs.json
    uv run python local_stage2_data.py --clean-data --data src/hgr/data/parallel.filtered.csv
"""
import argparse
import json
import logging
import os
import random
import re

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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


def main():
    parser = argparse.ArgumentParser(description="Generate DPO triplets from Stage 1 adapter")
    parser.add_argument("--data", default="src/hgr/data/parallel.csv")
    parser.add_argument("--lang-info", default="src/hgr/data/dataset_info.json")
    parser.add_argument("--lora", default="stage1_output")
    parser.add_argument("--output", default="dpo_pairs.json")
    parser.add_argument("--model", default="google/mt5-large")
    parser.add_argument("--sample", type=int, default=30000, help="Max pairs to generate")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--clean-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.lang_info, "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    src_lang = lang_data["src_lang"]
    tgt_lang = lang_data["tgt_lang"]

    if args.clean_data:
        args.data = "src/hgr/data/parallel.filtered.csv"
        logger.info("Using cleaned dataset (--clean-data): parallel.filtered.csv")

    logger.info(f"Loading {src_lang} <-> {tgt_lang} from {args.data}")
    df = pd.read_csv(args.data, encoding="utf-8")
    df["src"] = df["src"].apply(clean_text)
    df["tgt"] = df["tgt"].apply(clean_text)
    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)]
    df = df.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)
    logger.info(f"Clean pairs: {len(df)}")

    prefix_src2tgt = f"translate {src_lang} to {tgt_lang}: "
    prefix_tgt2src = f"translate {tgt_lang} to {src_lang}: "

    pairs = []
    for _, row in df.iterrows():
        pairs.append({"source": prefix_src2tgt + row["src"], "preferred": row["tgt"]})
        pairs.append({"source": prefix_tgt2src + row["tgt"], "preferred": row["src"]})

    random.shuffle(pairs)
    total = min(args.sample, len(pairs))
    pairs = pairs[:total]
    logger.info(f"Generating unpreferred translations for {len(pairs)} pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )

    lora_path = os.path.join(args.lora, "adapter_model.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(
            f"LoRA weights not found at {args.lora}/. Run train_stage1.py first."
        )

    model = PeftModel.from_pretrained(base_model, args.lora)
    model.eval()

    logger.info(
        f"Generating with temperature={args.temperature}, top_p={args.top_p}"
    )
    for i in tqdm(range(0, len(pairs), args.batch)):
        batch = pairs[i : i + args.batch]
        sources = [p["source"] for p in batch]

        inputs = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=args.max_len,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                early_stopping=True,
            )

        preds = [p.strip() for p in tokenizer.batch_decode(out, skip_special_tokens=True)]
        for pred, pair in zip(preds, batch):
            pair["unpreferred"] = pred

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(pairs)} DPO triplets to {args.output}")


if __name__ == "__main__":
    main()
