#!/usr/bin/env bash
# Everything that scores on the 2023 dev weeks, in order, on the current code.
# Never touches the 2024-25 test seasons.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
mkdir -p outputs/logs  # outputs/ is git-ignored, so a fresh clone has no log folder
L=outputs/logs
uv run serein backtest --tag dev > $L/backtest_dev.log 2>&1 && echo "dev done"
uv run python -m serein.eval.figures outputs/backtest/west_us_dev > $L/figures_dev.log 2>&1 || echo "FIGURES FAILED: see $L/figures_dev.log"
uv run python -m serein.eval.impact outputs/backtest/west_us_dev > outputs/backtest/west_us_dev/impact.txt 2>&1 || echo "IMPACT FAILED"
uv run serein map --date 2023-08-18 --run west_us_dev > $L/map_dev.log 2>&1 || echo "MAP FAILED: see $L/map_dev.log"
uv run serein backtest --tag dev_holdout --holdout-sites 0.2 --no-gbm > $L/backtest_dev_holdout.log 2>&1 && echo "holdout done"
./scripts/run_ablations.sh
echo ALLDONE
