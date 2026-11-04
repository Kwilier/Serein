#!/usr/bin/env bash
# Copy the latest figures into website/assets (TODO E-6). Re-run after the final test,
# point RUN at the final folder, and update the numbers in website/index.html.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-outputs/backtest/west_us_dev}
mkdir -p website/assets
cp "$RUN"/fig_rmse_by_lead_all.png "$RUN"/fig_improvement_map_L1.png website/assets/
cp "$RUN"/fig_case_*_L1.png website/assets/ 2>/dev/null || true
cp outputs/maps/map_20230818.png outputs/airplan/airplan_learning_curve.png website/assets/
echo "website/assets updated from $RUN"
