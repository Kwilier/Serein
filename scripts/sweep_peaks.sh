#!/usr/bin/env bash
# B-13: does up-weighting smoke days in training fix under-predicted peaks? Dev weeks only.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
mkdir -p outputs/logs  # outputs/ is git-ignored, so a fresh clone has no log folder
for w in 1 3 6; do
  uv run serein backtest --no-gbm --no-ablation --seeds 3 --smoke-weight $w --tag "peaks_w$w" \
    > "outputs/logs/peaks_w$w.log" 2>&1
  echo "done w=$w"
done
