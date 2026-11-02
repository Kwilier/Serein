#!/usr/bin/env bash
# Everything that scores on the 2023 dev weeks, in order, on the current code.
# Never touches the 2024-25 test seasons.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
L=outputs/logs
uv run serein backtest --tag dev > $L/backtest_dev.log 2>&1 && echo "dev done"
uv run python -m serein.eval.figures outputs/backtest/west_us_dev > /dev/null 2>&1
uv run python -m serein.eval.impact outputs/backtest/west_us_dev > outputs/backtest/west_us_dev/impact.txt 2>&1
uv run serein map --date 2023-08-18 --run west_us_dev > /dev/null 2>&1
uv run serein backtest --tag dev_holdout --holdout-sites 0.2 --no-gbm > $L/backtest_dev_holdout.log 2>&1 && echo "holdout done"
./scripts/run_ablations.sh
echo ALLDONE
