# HGR — Low-Resource Machine Translation with DPO + Hypergeometric-Gamma Reward

An implementation of the two-stage training pipeline for enhancing low-resource language machine translation using Direct Preference Optimization (DPO) combined with a novel Hypergeometric-Gamma Reward (HGR) function, built on top of mT5-large.

## Pipeline Overview

The pipeline consists of three stages executed sequentially:

### Stage 1: Supervised Fine-Tuning (SFT)

Fine-tunes `mT5-large` on parallel sentence pairs using LoRA adapters. The model learns bidirectional translation (source-to-target and target-to-source) using task prefixes.

- Input: `src/hgr/data/parallel.csv` (parallel corpus) + `dataset_info.json` (language metadata)
- Output: LoRA adapter weights in `stage1_output/`
- Run: `modal run --detach modal_stage1.py`

### Stage 2a: DPO Data Generation

Generates "unpreferred" (intentionally imperfect) translations from the Stage 1 model using high-temperature sampling. These are paired with reference translations to create DPO training triplets.

- Input: Stage 1 adapter weights + parallel corpus
- Output: `stage2_output/dpo_dataset_30k_sampled.json`
- Run: `uv run python local_stage2_data.py` (local) or `modal run --detach modal_stage2_data.py`

### Stage 2b: Combined DPO + HGR Training

Trains the model using a combined objective:

- **DPO Loss** (Eq. 1): Maximizes the gap between preferred and unpreferred translation log-probabilities relative to a frozen reference model.
- **HGR Loss** (Eq. 2-4): Uses Sentence-BERT cosine similarity to compute a hypergeometric-gamma reward `r = rho * exp(-phi * rho)`, then applies REINFORCE-style policy gradient.
- **Combined Loss** (Eq. 5): `L = alpha * L_DPO + gamma * L_HGR`
- **Exponential Gradient Clipping** (EGC): Prevents gradient explosion in later epochs.

Run: `modal run --detach modal_stage2_train.py`

## Evaluation

Metrics implemented in `src/hgr/evaluation/metrics.py`:

- BLEU (SacreBLEU)
- chrF++
- METEOR
- BERTScore
- Approximate Randomization Test (statistical significance)
- Cohen's d (effect size)

Run: `uv run python evaluate_stage2.py`

## Quick Start

```bash
# Install dependencies
uv sync

# 1. Place your parallel corpus at src/hgr/data/parallel.csv (columns: src, tgt)
# 2. Create src/hgr/data/dataset_info.json with {"src_lang": "...", "tgt_lang": "..."}

# 3. Stage 1 — SFT
modal run --detach modal_stage1.py

# 4. Stage 2a — Generate DPO data
uv run python local_stage2_data.py

# 5. Stage 2b — DPO+HGR training
modal run --detach modal_stage2_train.py

# 6. Evaluate
uv run python evaluate_stage2.py
```

See [docs/PIPELINE.md](docs/PIPELINE.md) for detailed documentation.

## Project Structure

```
src/hgr/
  config.py           -- ModelConfig, TrainingConfig, EvalConfig dataclasses
  data/prepare.py     -- Data loading and unpreferred translation generation
  training/dpo.py     -- TRL DPOTrainer wrapper (Algorithm 1)
  training/hgr.py     -- HGR reward computation and REINFORCE loss (Algorithm 2)
  training/combined.py-- Combined DPO+HGR training loop (Algorithm 3)
  evaluation/metrics.py -- MT evaluation metrics and significance tests
```

## References

- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023
- Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer", NAACL 2021
- Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
- Fan et al., "Beyond English-Centric Multilingual Machine Translation" (M2M-100), JMLR 2021
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
