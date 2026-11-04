#!/usr/bin/env bash
# Everything that scores on the 2023 dev weeks, in order, on the current code.
# Never touches the 2024-25 test seasons.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
mkdir -p outputs/logs  # outputs/ is git-ignored, so a fresh clone has no log folder
L=outputs/logs
run_logged () {
    local label=$1 log_path=$2 status
    shift 2
    if "$@" > "$log_path" 2>&1; then
        echo "$label done"
    else
        status=$?
        echo "$label FAILED (exit $status): see $log_path" >&2
        return "$status"
    fi
}
R=outputs/backtest/west_us_dev
mkdir -p "$R"
run_logged dev "$L/backtest_dev.log" uv run serein backtest --tag dev
run_logged figures "$L/figures_dev.log" uv run python -m serein.eval.figures "$R"
run_logged impact "$R/impact.txt" uv run python -m serein.eval.impact "$R"
run_logged map "$L/map_dev.log" uv run serein map --date 2023-08-18 --run west_us_dev
run_logged holdout "$L/backtest_dev_holdout.log" uv run serein backtest --tag dev_holdout --holdout-sites 0.2 --no-gbm
./scripts/run_ablations.sh
echo ALLDONE
