# Autoresearch Session: BLEU Optimization (Modal)

- Metric: **BLEU** (higher is better)
- Unit: score
- Benchmark command: `uvx --from modal modal run modal_benchmark_bleu.py`
- Files in scope: all files in repository
- Environment: Modal (cost-aware, A10G)
- Max iterations: 20
- Git policy: commit + push at significant steps

## Notes
`pi-autoresearch` CLI subcommands (`init_experiment`, `run_experiment`, `log_experiment`) were not available in PATH, so this session uses an equivalent manual loop with durable logs.

## Iteration Ledger

| Iter | Change | BLEU | Keep? | Commit |
|---:|---|---:|---|---|
| 0 | Baseline (max_len=48, num_beams=1) | 2.3526 | yes | pending |
| 1 | num_beams=4 @ max_len=48 | 2.3430 | no (reverted) | pending |
| 2 | max_len=64, num_beams=1 | 2.8155 | yes | pending |
| 3 | max_len=96 | 2.5885 | no (reverted) | pending |
| 4 | num_beams=2 @ max_len=64 | 2.8368 | yes | pending |
| 5 | num_beams=3 @ max_len=64 | 2.9293 | yes (best) | pending |
| 6 | num_beams=4 @ max_len=64 | 2.8361 | no (reverted) | pending |
