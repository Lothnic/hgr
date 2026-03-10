import json
import random

def main():
    print("Loading original dataset...")
    # Load the 109,000 generated DPO pairs from Stage 1
    with open("stage2_output_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("src/hgr/data/dataset_info.json", "r", encoding="utf-8") as f:
        lang_data = json.load(f)
    src_lang = lang_data["src_lang"]
    tgt_lang = lang_data["tgt_lang"]

    # We want exactly 10,000 pairs sampled randomly
    src2tgt_pairs = [d for d in data if f"{src_lang} to {tgt_lang}" in d["source"]]
    tgt2src_pairs = [d for d in data if f"{tgt_lang} to {src_lang}" in d["source"]]

    print(f"Original {src_lang}->{tgt_lang}: {len(src2tgt_pairs)}")
    print(f"Original {tgt_lang}->{src_lang}: {len(tgt2src_pairs)}")

    random.seed(42)  # For reproducibility
    sampled_src2tgt = random.sample(src2tgt_pairs, min(5000, len(src2tgt_pairs)))
    sampled_tgt2src = random.sample(tgt2src_pairs, min(5000, len(tgt2src_pairs)))

    downsampled_dataset = sampled_src2tgt + sampled_tgt2src
    # Shuffle the final combined list
    random.shuffle(downsampled_dataset)

    print(f"Saving downsampled dataset with {len(downsampled_dataset)} pairs...")
    with open("dpo_dataset_10k.json", "w", encoding="utf-8") as f:
        json.dump(downsampled_dataset, f, ensure_ascii=False, indent=2)

    print("Done! Saved as `dpo_dataset_10k.json`")

if __name__ == "__main__":
    main()
