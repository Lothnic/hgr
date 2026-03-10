import json
from src.hgr.evaluation.metrics import evaluate_all

def main():
    print("Loading data...")
    with open("stage2_output", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("src/hgr/data/dataset_info.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    src_lang = lang_data["src_lang"]
    tgt_lang = lang_data["tgt_lang"]

    # Separate by direction
    src2tgt_preds, src2tgt_refs = [], []
    tgt2src_preds, tgt2src_refs = [], []

    for item in data:
        src = item["source"]
        pred = item["unpreferred"]
        ref = item["preferred"]

        if f"{src_lang} to {tgt_lang}" in src:
            src2tgt_preds.append(pred)
            src2tgt_refs.append(ref)
        elif f"{tgt_lang} to {src_lang}" in src:
            tgt2src_preds.append(pred)
            tgt2src_refs.append(ref)

    print(f"Total pairs: {len(data)}")
    print(f"S→T pairs: {len(src2tgt_refs)}")
    print(f"T→S pairs: {len(tgt2src_refs)}")

    print(f"\n--- Benchmarking {src_lang}→{tgt_lang} ---")
    if src2tgt_preds:
        src2tgt_metrics = evaluate_all(src2tgt_preds, src2tgt_refs, metrics=["bleu", "chrf"])
        print(f"BLEU: {src2tgt_metrics['bleu']:.2f}")
        print(f"chrF++: {src2tgt_metrics['chrf']:.2f}")

    print(f"\n--- Benchmarking {tgt_lang}→{src_lang} ---")
    if tgt2src_preds:
        tgt2src_metrics = evaluate_all(tgt2src_preds, tgt2src_refs, metrics=["bleu", "chrf"])
        print(f"BLEU: {tgt2src_metrics['bleu']:.2f}")
        print(f"chrF++: {tgt2src_metrics['chrf']:.2f}")

    print("\n--- Benchmarking Overall ---")
    all_preds = src2tgt_preds + tgt2src_preds
    all_refs = src2tgt_refs + tgt2src_refs
    overall_metrics = evaluate_all(all_preds, all_refs, metrics=["bleu", "chrf"])
    print(f"BLEU: {overall_metrics['bleu']:.2f}")
    print(f"chrF++: {overall_metrics['chrf']:.2f}")

if __name__ == "__main__":
    main()
