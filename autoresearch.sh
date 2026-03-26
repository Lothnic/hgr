#!/usr/bin/env bash
set -euo pipefail

CMD="uvx --from modal modal run modal_benchmark_bleu.py"

echo "[autoresearch] Running benchmark: $CMD"
OUT=$($CMD 2>&1 | tee /tmp/autoresearch_last_run.log)
BLEU=$(echo "$OUT" | grep -Eo 'FINAL_BLEU=[0-9]+(\.[0-9]+)?' | tail -n1 | cut -d= -f2)

if [[ -z "${BLEU:-}" ]]; then
  echo "[autoresearch] ERROR: Could not parse FINAL_BLEU"
  exit 1
fi

echo "[autoresearch] BLEU=$BLEU"
