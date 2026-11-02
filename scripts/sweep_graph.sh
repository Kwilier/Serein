#!/usr/bin/env bash
# Dev-mode ablation sweep (B-10): does message passing help, and under which settings?
# Scores ONLY on the 2023 calibration weeks. Never touches the test seasons.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run () { tag=$1; shift; uv run serein backtest --no-gbm --tag "sweep_$tag" "$@" > "outputs/logs/sweep_$tag.log" 2>&1
         echo "== $tag $*"; sed -n '/^RMSE (/,/^$/p;/^RMSE at HIDDEN/,/^$/p' "outputs/logs/sweep_$tag.log" | grep -E "RMSE|lead|smokecast|noaa_best"; }
REG="--dropout 0.3 --weight-decay 1e-3"
run mask0_reg   --node-mask 0 $REG
run small_reg   --hidden 64 --layers 2 $REG
run reg_seed1   $REG --seed 1
run holdout_reg $REG --holdout-sites 0.2
run ens5_reg    $REG --seeds 5
