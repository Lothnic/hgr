# HGR — Low-Resource Machine Translation with DPO + Hypergeometric-Gamma Reward

An implementation of the two-stage training pipeline for enhancing low-resource language machine translation using Direct Preference Optimization (DPO) combined with a novel Hypergeometric-Gamma Reward (HGR) function, built on top of mT5.

## Pipeline Overview

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tunes mT5 on parallel sentence pairs using LoRA adapters. The model learns bidirectional translation (source-to-target and target-to-source) using task prefixes.

```bash
uv run python train_stage1.py
uv run python train_stage1.py --model google/mt5-base --epochs 5 --batch 32
```

- Input: `src/hgr/data/parallel.csv` + `dataset_info.json`
- Output: LoRA adapter weights in `stage1_output/`

### Stage 2a: DPO Data Generation

Generates "unpreferred" translations from the Stage 1 model using high-temperature sampling.

```bash
uv run python local_stage2_data.py --lora stage1_output
uv run python main.py gen-unpreferred --use-stage1 --input src/hgr/data/parallel.csv
```

- Input: Stage 1 adapter + parallel corpus
- Output: DPO triplets JSON (source, preferred, unpreferred)

### Stage 2b: Training (DPO / HGR / Combined)

```bash
uv run python main.py train --method dpo      --data dpo_pairs.json
uv run python main.py train --method hgr      --data src/hgr/data/parallel.csv
uv run python main.py train --method combined --data dpo_pairs.json --reward hgr
```

- **DPO Loss**: Maximizes gap between preferred and unpreferred log-probabilities vs a frozen reference model.
- **HGR Loss**: SBERT cosine similarity → hypergeometric-gamma reward `r = ρ · exp(-φ · ρ)` → REINFORCE policy gradient.
- **Combined Loss**: `L = α · L_DPO + γ · L_HGR`
- **Exponential Gradient Clipping**: Prevents gradient explosion in later epochs.

### Evaluation

```bash
uv run python main.py evaluate --predictions preds.txt --references refs.txt
```

Metrics: BLEU (SacreBLEU), chrF++, METEOR, BERTScore, Approximate Randomization Test, Cohen's d.

## Quick Start

```bash
uv sync

# Place your parallel corpus at src/hgr/data/parallel.csv (columns: src, tgt)
# Create src/hgr/data/dataset_info.json with {"src_lang": "...", "tgt_lang": "..."}

# Clean data (optional)
uv run python scripts/clean_parallel_data.py

# Stage 1 — SFT
uv run python train_stage1.py

# Stage 2a — Generate DPO triplets
uv run python local_stage2_data.py

# Stage 2b — Train
uv run python main.py train --method combined --data dpo_pairs.json

# Evaluate
uv run python main.py evaluate --predictions preds.txt --references refs.txt
```

## Data Cleaning

```bash
uv run python scripts/clean_parallel_data.py --input src/hgr/data/parallel.csv --output src/hgr/data/parallel.filtered.csv
```

Filters: empty pairs, exact matches, duplicates, length-ratio outliers, artifact noise, heavy Latin character leakage.

## Project Structure

```
├── main.py                  # CLI: prepare-data, gen-unpreferred, train, evaluate
├── train_stage1.py          # Stage 1 SFT training (local GPU)
├── local_stage2_data.py     # Stage 2a DPO triplet generation (local GPU)
├── scripts/
│   └── clean_parallel_data.py
└── src/hgr/
    ├── config.py            # ModelConfig, TrainingConfig, RewardConfig, EvalConfig
    ├── data/
    │   ├── prepare.py       # Data loading utilities
    │   ├── parallel.csv     # Parallel corpus
    │   └── dataset_info.json
    ├── training/
    │   ├── dpo.py           # TRL DPOTrainer wrapper (Algorithm 1)
    │   ├── hgr.py           # HGR reward + REINFORCE loss (Algorithm 2)
    │   └── combined.py      # Combined DPO+HGR training (Algorithm 3)
    ├── evaluation/
    │   └── metrics.py       # MT evaluation metrics + significance tests
    └── rewards/
        ├── base.py          # Reward function ABC
        ├── factory.py       # Reward factory
        ├── hgr_reward.py    # SBERT cosine similarity → HGR reward
        ├── bleurt_reward.py # BLEURT-based reward
        └── comet_reward.py  # COMET-based reward
```

## References

- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023
- Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer", NAACL 2021
- Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
- Fan et al., "Beyond English-Centric Multilingual Machine Translation" (M2M-100), JMLR 2021
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
