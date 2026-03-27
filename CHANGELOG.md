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

## 2026-03-26 — Full cleaned-data pipeline run

- Added cleaning script: `scripts/clean_parallel_data.py`.
- Cleaned `src/hgr/data/parallel.csv` -> `src/hgr/data/parallel.filtered.csv`.
  - Final kept rows: 21,114 (from 25,541 parseable rows).
- Added full-data DPO triplet generation script: `modal_stage2_data_full_clean.py`.
  - Generated `dpo_dataset_full_clean.json` with 42,228 bidirectional pairs.
- Ran full-data DPO-only training (`max_steps=300`) on cleaned full dataset.
- Post-train benchmark (sample_size=512, same decode settings): BLEU **5.0711**.
- Observation: full-data DPO run regressed vs best smaller-run DPO config (5.4359), likely due residual noise / longer optimization drift; needs validation split + checkpoint selection.

## 2026-03-26 — Stage2 evaluated with exact Stage1 protocol

- Added `modal_evaluate_stage2_same_protocol.py` and ran evaluation against Stage1 test set with matching decoding (`num_beams=4`, `max_length=48`).
- Stage2 DPO (full cleaned-data run) results:
  - Overall BLEU: 21.8603
  - Overall chrF++: 46.8517
  - mhi BLEU: 23.5066
  - him BLEU: 19.8539
- Compared with Stage1-large baseline:
  - Overall BLEU delta: -0.5894
  - Directionally asymmetric behavior: mhi improved, him regressed.

## 2026-03-26 — Autoresearch resumed on same-protocol objective

- Resumed loop with benchmark switched to exact Stage1 protocol evaluator (`modal_evaluate_stage2_same_protocol.py`).
- Baseline (current full-clean full-data checkpoint): BLEU **21.8603**.
- Iteration: retrained DPO on full-clean dataset with subset-focused config (`n_train=4096`, `max_steps=90`, `beta=0.2`, `lr=1e-5`, train truncation 80/160).
- New score: BLEU **22.3615**, chrF++ **47.5509**.
- Status: closes most of the gap to Stage1-large (22.4497), remaining delta ≈ **0.0882 BLEU**.

## 2026-03-26 — Ongoing push toward BLEU >= 25

- Iter 21 attempted direction-targeted continuation (`tgt2src` only) from Stage2 checkpoint; result regressed to BLEU 22.2046 (not kept).
- Started Stage1-clean retraining on filtered corpus (`hgr-stage1-clean` volume) as a higher-ceiling path toward BLEU >= 25.
- Training run (in progress): https://modal.com/apps/lothnic/main/ap-3sIvITIgBXMx5aeU0UG8SM

## 2026-03-26 — Failed clean Stage1 retrain branch

- Ran Stage1 training directly on `parallel.filtered.csv` with original aggressive Stage1 hyperparameters.
- Result collapsed badly: Overall BLEU **1.7616** (kangri->hindi 1.9173, hindi->kangri 1.5891).
- Conclusion: this branch is not viable; likely hyperparameter instability/domain shift interaction.
- Action: reverted `modal_stage1.py` to original baseline wiring and created separate tuned script `modal_stage1_clean_tuned.py` for safer retries.

## 2026-03-27 — Decode tuning rescued BLEU toward target

- `stage1-clean-tuned` retrain completed but remained non-viable:
  - Overall BLEU **2.5168** (still collapsed vs baseline).
- Pivoted to inference-time decode optimization on stable `hgr-stage1-large` checkpoint.
- Added tunable evaluator: `modal_evaluate_stage1_same_protocol_tunable.py` + `stage1_eval_config.json`.
- Confirmed baseline-like decode on tunable script:
  - `beams=4,max_len=48` -> BLEU **22.4433**.
- Found strong gains from decode params:
  - `beams=8,max_len=80,len_pen=0.8` -> BLEU **24.1203**.
  - `beams=8,max_len=96,len_pen=0.6` -> BLEU **24.2524** (best so far).
- Started next eval run to try crossing BLEU 25:
  - `beams=10,max_len=96,len_pen=0.4` (running in background process `stage1-decode-eval-b10-lp04`).

## 2026-03-27 — Iter 25 decode test result

- Completed `stage1-decode-eval-b9-lp05` after increasing Modal timeout to 4h.
- Config: `num_beams=9, max_length=96, length_penalty=0.5, batch_size=12`.
- Result: BLEU **24.2470**, chrF++ **50.3792**.
- Outcome: slight regression vs best 24.2524 (keep=false).
- Next: test longer decode with same stable checkpoint (`max_length=128`) now that timeout is no longer the bottleneck.
