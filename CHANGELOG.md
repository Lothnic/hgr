# CHANGELOG

## 2026-03-26 — Autoresearch loop started (Modal BLEU optimization)

- Initialized autoresearch artifacts: `autoresearch.md`, `autoresearch.sh`, `autoresearch.jsonl`, `autoresearch_config.json`.
- Added `modal_benchmark_bleu.py` for cost-aware BLEU benchmarking on Modal A10G.
- Fixed benchmark blockers:
  - Robust CSV parsing (`engine='python'`, `on_bad_lines='skip'`) for malformed lines.
  - Adapter/base-model mismatch by auto-reading `/stage1_output/adapter_config.json`.
- Baseline: BLEU **2.3526** (`max_len=48`, `num_beams=1`).
- Best so far: BLEU **2.9293** (`max_len=64`, `num_beams=3`).
- Failed candidates:
  - `num_beams=4` at `max_len=48` (BLEU 2.3430)
  - `max_len=96` (BLEU 2.5885)
  - `num_beams=4` at `max_len=64` (BLEU 2.8361)
- Next step: continue search over decoding settings and then move from decode-only optimization to training/reward-side code paths.

## 2026-03-26 — Iteration milestone

- Explored additional decode settings after initial best.
- `do_sample=true, temperature=0.7, top_p=0.9` regressed to BLEU **2.7836** (reverted).
- Re-ran best deterministic config (`max_len=64`, `num_beams=3`) and got BLEU **2.9295**.
- Current best config in `autoresearch_config.json` is deterministic and cost-aware.
- Next step: pivot from decoding-only tuning to source-code fixes in training/data pipeline while keeping this benchmark as a regression guard.

## 2026-03-26 — 10-iteration deep dive on low BLEU causes (Iter 9–18)

- Completed 10 additional iterations focused on BLEU recovery + root-cause analysis.
- Major fix: switched benchmark model volume from `hgr-stage1` to `hgr-stage1-large`.
  - `hgr-stage1` adapter config indicates `task_type=SEQ_CLS` (not seq2seq generation-oriented).
  - `hgr-stage1-large` adapter config indicates `task_type=SEQ_2_SEQ_LM`.
  - BLEU jump observed: from ~2.93 to **3.87** immediately.
- Decoding sweeps on `hgr-stage1-large`:
  - Kept: `repetition_penalty=1.1`, `length_penalty=0.8`, `max_len=80`.
  - Rejected: `no_repeat_ngram_size=3`, `length_penalty` at 0.6/1.2, `max_len=96`.
- Best score this run: **BLEU 5.2567** with `sample_size=512`, `max_len=80`, `num_beams=3`, `rep_penalty=1.1`, `len_penalty=0.8`.
- Data-quality diagnostics executed:
  - 26,785 lines total vs 25,541 parseable rows → **1,244 malformed rows**.
  - ~**13.01%** noisy-pattern rows; ~**9.26%** severe length-ratio outliers.
- Next step: move from decode-only tuning to dataset cleaning + retraining on verified aligned pairs; decoding gains are now incremental.

## 2026-03-26 — DPO-only 10-iteration ablation (with per-iteration pushes)

- Implemented `modal_stage2_dpo_train.py` (manual DPO objective, no reward/penalty term).
- Extended `modal_benchmark_bleu.py` to evaluate `hgr-stage2-dpo` model volume.
- Ran 10 DPO-only iterations on Modal A10G, pushing each iteration commit.
- Best DPO-only run in this block: **BLEU 5.4359** (iter 9; `lr=1e-5`, `beta=0.2`, `max_steps=90`, `n_train=4096`, train truncation 80/160).
- Baseline DPO-only run in this block: **BLEU 5.2573** (iter 1).
- Net improvement across this block: **+0.1786 BLEU**.
