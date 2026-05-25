"""
HGR Machine Translation — CLI Entry Point

Usage:
    # Load and inspect your parallel corpus
    uv run python main.py prepare-data --input src/hgr/data/parallel.csv

    # Generate unpreferred translations via Stage1 PEFT adapter
    uv run python main.py gen-unpreferred --input src/hgr/data/parallel.csv --lora stage1_output

    # Train
    uv run python main.py train --method dpo      --data dpo_pairs.json
    uv run python main.py train --method hgr      --data src/hgr/data/parallel.csv
    uv run python main.py train --method combined --data dpo_pairs.json

    # Evaluate
    uv run python main.py evaluate --predictions preds.txt --references refs.txt
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hgr")


def cmd_prepare_data(args):
    from hgr.data.prepare import load_parallel_data

    ds = load_parallel_data(args.input, src_col=args.src_col, tgt_col=args.tgt_col)
    print(f"Loaded {len(ds)} sentence pairs")
    print(f"Columns: {ds.column_names}")
    print(f"Sample: {ds[0]}")


def cmd_gen_unpreferred(args):
    from hgr.data.prepare import load_parallel_data, generate_unpreferred

    if args.use_stage1:
        import json
        import os
        import re

        import pandas as pd
        import torch
        from peft import PeftModel
        from tqdm import tqdm

        with open(args.lang_info, "r", encoding="utf-8") as f:
            lang = json.load(f)
        src_lang = lang["src_lang"]
        tgt_lang = lang["tgt_lang"]

        def clean(x):
            if not isinstance(x, str):
                return ""
            return re.sub(r"\s+", " ", x.strip().strip("\"'"))

        df = pd.read_csv(args.input, encoding="utf-8")
        df["src"] = df["src"].apply(clean)
        df["tgt"] = df["tgt"].apply(clean)
        df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)]
        df = df.drop_duplicates(subset=["src", "tgt"]).reset_index(drop=True)

        prefix_fwd = f"translate {src_lang} to {tgt_lang}: "
        prefix_rev = f"translate {tgt_lang} to {src_lang}: "

        pairs = []
        for _, row in df.iterrows():
            pairs.append({"source": prefix_fwd + row["src"], "preferred": row["tgt"]})
            pairs.append({"source": prefix_rev + row["tgt"], "preferred": row["src"]})

        if args.sample > 0:
            import random
            random.Random(args.seed).shuffle(pairs)
            pairs = pairs[: min(args.sample, len(pairs))]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.stage1_model)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.stage1_model,
            torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )

        lora_check = os.path.join(args.lora, "adapter_model.safetensors")
        if not os.path.exists(lora_check):
            raise FileNotFoundError(f"LoRA weights not found in {args.lora}")

        model = PeftModel.from_pretrained(base_model, args.lora)
        model.eval()

        print(f"Generating {len(pairs)} unpreferred translations via Stage1 adapter...")
        for i in tqdm(range(0, len(pairs), args.batch_size)):
            batch = pairs[i : i + args.batch_size]
            sources = [p["source"] for p in batch]
            inputs = tokenizer(
                sources, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_len,
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_length=args.max_len,
                    do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, early_stopping=True,
                )
            preds = [p.strip() for p in tokenizer.batch_decode(out, skip_special_tokens=True)]
            for pred, pair in zip(preds, batch):
                pair["unpreferred"] = pred

        output = args.output or args.input.replace(".csv", "_dpo.json")
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(pairs)} DPO triplets to {output}")
    else:
        ds = load_parallel_data(args.input, src_col=args.src_col, tgt_col=args.tgt_col)
        dpo_ds = generate_unpreferred(
            ds,
            model_name=args.model,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            batch_size=args.batch_size,
        )
        output = args.output or args.input.replace(".csv", "_dpo.json")
        dpo_ds.to_json(output)
        print(f"Saved DPO dataset ({len(dpo_ds)} triplets) to {output}")


def cmd_train(args):
    from hgr.config import ModelConfig, TrainingConfig

    if args.method == "dpo":
        from hgr.training.dpo import build_dpo_trainer, prepare_dpo_dataset
        import json
        from datasets import Dataset as HFDataset

        with open(args.data, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ds = HFDataset.from_list(raw)
        dpo_ds = prepare_dpo_dataset(ds)
        trainer = build_dpo_trainer(train_dataset=dpo_ds)
        trainer.train()

    elif args.method == "hgr":
        from hgr.training.hgr import HGRTrainer

        cfg = TrainingConfig(learning_rate=args.lr, batch_size=args.batch)
        trainer = HGRTrainer(
            args.model or ModelConfig().base_model,
            sbert_model_name="sentence-transformers/all-MiniLM-L6-v2",
            training_config=cfg,
        )
        from datasets import Dataset as HFDataset
        trainer.train(
            HFDataset.from_csv(args.data),
            num_epochs=args.epochs,
            batch_size=args.batch,
        )

    elif args.method == "combined":
        from hgr.training.combined import CombinedTrainer
        from hgr.config import RewardConfig
        import json
        from datasets import Dataset as HFDataset

        cfg = TrainingConfig(
            learning_rate=args.lr,
            batch_size=args.batch,
            dpo_beta=args.dpo_beta,
            alpha=args.alpha,
            gamma=1.0 - args.alpha,
        )
        reward_cfg = RewardConfig(reward_function=args.reward)
        trainer = CombinedTrainer(
            ModelConfig().base_model,
            reward_config=reward_cfg,
            training_config=cfg,
        )
        with open(args.data, "r", encoding="utf-8") as f:
            raw = json.load(f)
        trainer.train(
            HFDataset.from_list(raw),
            num_epochs=args.epochs,
            batch_size=args.batch,
        )


def cmd_evaluate(args):
    from hgr.evaluation.metrics import evaluate_all

    with open(args.predictions) as f:
        preds = [line.strip() for line in f]
    with open(args.references) as f:
        refs = [line.strip() for line in f]
    results = evaluate_all(preds, refs)
    print("\n=== Evaluation Results ===")
    for metric, score in results.items():
        print(f"  {metric:>12s}: {score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="HGR Machine Translation Framework")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare-data", help="Load and inspect your parallel corpus")
    p.add_argument("--input", required=True)
    p.add_argument("--src-col", default="source")
    p.add_argument("--tgt-col", default="target")

    p = sub.add_parser("gen-unpreferred", help="Generate unpreferred translations for DPO")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--use-stage1", action="store_true", help="Use Stage1 PEFT adapter (recommended)")
    p.add_argument("--lora", default="stage1_output", help="Path to Stage1 LoRA adapter")
    p.add_argument("--stage1-model", default="google/mt5-large")
    p.add_argument("--lang-info", default="src/hgr/data/dataset_info.json")
    p.add_argument("--model", default="facebook/m2m100_418M", help="Fallback model (without --use-stage1)")
    p.add_argument("--src-lang", default="hi")
    p.add_argument("--tgt-lang", default="en")
    p.add_argument("--src-col", default="source")
    p.add_argument("--tgt-col", default="target")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=48)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--sample", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train", help="Train with DPO, HGR, or combined")
    p.add_argument("--method", choices=["dpo", "hgr", "combined"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--dpo-beta", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--reward", default="hgr", choices=["hgr", "bleurt", "comet"])

    p = sub.add_parser("evaluate", help="Evaluate translations")
    p.add_argument("--predictions", required=True)
    p.add_argument("--references", required=True)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "prepare-data": cmd_prepare_data,
        "gen-unpreferred": cmd_gen_unpreferred,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
