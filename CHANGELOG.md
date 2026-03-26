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
