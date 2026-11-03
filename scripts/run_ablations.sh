#!/usr/bin/env bash
# B-10: which inputs matter? Each run hides one input group from SmokeCast (and from
# its base forecast where relevant). Dev weeks only; 3-seed ensembles.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
mkdir -p outputs/logs  # outputs/ is git-ignored, so a fresh clone has no log folder
L=outputs/logs
abl () { tag=$1; shift; uv run serein backtest --no-gbm --no-ablation --seeds 3 --tag "abl_$tag" "$@" > "$L/abl_$tag.log" 2>&1; echo "abl $tag done"; }
abl full
abl no_wind    --no-wind
abl no_hrrr    --drop hrrr_,wind_now_
abl no_aqm     --drop aqm_
abl no_fires   --drop frp_,fire_dist
abl no_hms     --drop hms_prev
abl no_sensors --drop obs_
echo ALLDONE
