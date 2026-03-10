"""
HGR Machine Translation — Entry Point

Usage:
    # Load and prep your custom dataset
    uv run python main.py prepare-data --input data/my_corpus.csv

    # Generate unpreferred translations for DPO pairs
    uv run python main.py gen-unpreferred --input data/my_corpus.csv --src-lang hi --tgt-lang en

    # Train (implement the TODOs in training/ first!)
    uv run python main.py train --method dpo        --data data/dpo_pairs.json
    uv run python main.py train --method hgr        --data data/my_corpus.csv
    uv run python main.py train --method combined   --data data/dpo_pairs.json

    # Evaluate
    uv run python main.py evaluate --predictions preds.txt --references refs.txt
"""
import sys
import argparse
import logging

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
    ds = load_parallel_data(args.input, src_col=args.src_col, tgt_col=args.tgt_col)
    dpo_ds = generate_unpreferred(
        ds,
        model_name=args.model,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
    )
    output = args.output or args.input.replace(".csv", "_dpo.json").replace(".jsonl", "_dpo.json")
    dpo_ds.to_json(output)
    print(f"Saved DPO dataset ({len(dpo_ds)} triplets) to {output}")


def cmd_train(args):
    print(f"Training with method: {args.method}")
    print(f"Data: {args.data}")
    print()
    if args.method == "dpo":
        print("→ Implement DPOTrainer in src/hgr/training/dpo.py")
    elif args.method == "hgr":
        print("→ Implement HGRTrainer in src/hgr/training/hgr.py")
    elif args.method == "combined":
        print("→ Implement CombinedTrainer in src/hgr/training/combined.py")
    print("\nSee the TODO comments and paper equation references in each file.")


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

    # prepare-data
    p = sub.add_parser("prepare-data", help="Load and inspect your parallel corpus")
    p.add_argument("--input", required=True, help="Path to CSV/JSON/JSONL or HF dataset ID")
    p.add_argument("--src-col", default="source", help="Source column name")
    p.add_argument("--tgt-col", default="target", help="Target column name")

    # gen-unpreferred
    p = sub.add_parser("gen-unpreferred", help="Generate unpreferred translations via m2m100")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--model", default="facebook/m2m100_418M")
    p.add_argument("--src-lang", default="hi", help="m2m100 source language code")
    p.add_argument("--tgt-lang", default="en", help="m2m100 target language code")
    p.add_argument("--src-col", default="source")
    p.add_argument("--tgt-col", default="target")
    p.add_argument("--batch-size", type=int, default=16)

    # train
    p = sub.add_parser("train", help="Train (implement TODOs first!)")
    p.add_argument("--method", choices=["dpo", "hgr", "combined"], required=True)
    p.add_argument("--data", required=True)

    # evaluate
    p = sub.add_parser("evaluate", help="Evaluate translations")
    p.add_argument("--predictions", required=True, help="File with one prediction per line")
    p.add_argument("--references", required=True, help="File with one reference per line")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"prepare-data": cmd_prepare_data, "gen-unpreferred": cmd_gen_unpreferred,
     "train": cmd_train, "evaluate": cmd_evaluate}[args.command](args)


if __name__ == "__main__":
    main()
