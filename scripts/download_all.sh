#!/usr/bin/env bash
# Full data pull for configs/west_us.yaml (TODO A-10). Safe to re-run: finished days are skipped.
# Takes ~2-3 hours on a fast connection; stores ~5 GB of extracted values.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
YEARS=${YEARS:-2021,2022,2023,2024,2025}
CFG=${CFG:-configs/west_us.yaml}
uv run serein download airnow --config "$CFG" --years "$YEARS" --workers 8
uv run serein download hms    --config "$CFG" --years "$YEARS" --workers 8
uv run serein build-points    --config "$CFG" --years "$YEARS"
uv run serein download aqm    --config "$CFG" --years "$YEARS" --workers 8
uv run serein download hrrr   --config "$CFG" --years "$YEARS" --workers 12
echo "DONE"
