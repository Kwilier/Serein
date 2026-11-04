"""NOAA Air Quality Model (NAQFC / AQM): NOAA's official 72-hour PM2.5 forecast.

Public on AWS (bucket ``noaa-nws-naqfc-pds``). Versions by date:
  AQMv5 (CMAQ) ... 2021-07-19, AQMv6 2021-07-20 ... 2024-05-13, AQMv7 2024-05-14 ...
Each cycle (06z, 12z) has an hourly file with 72 one-hour averages:
  aqm.t12z.ave_1hr_pm25_bc.YYYYMMDD.227.grib2   (bias-corrected, NOAA's best)
  aqm.t12z.ave_1hr_pm25.YYYYMMDD.227.grib2      (raw model)

We do NOT use the ready-made 24 h average files: their averaging window is not
the local calendar day, so they can't be compared fairly with observations.
A message with steps (s, s+1) is the mean over [cycle+s, cycle+s+1) UTC.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from serein.config import interim_dir
from serein.data.grib import PointIndex, decode_messages
from serein.http import get

BUCKET = "https://noaa-nws-naqfc-pds.s3.amazonaws.com"
VERSIONS = [  # (first day, prefix)
    (date(2024, 5, 14), "AQMv7"),
    (date(2021, 7, 20), "AQMv6"),
    (date(2000, 1, 1), "AQMv5"),
]


def candidate_urls(d: date, cycle: int, bias_corrected: bool) -> list[str]:
    suffix = "_bc" if bias_corrected else ""
    name = f"aqm.t{cycle:02d}z.ave_1hr_pm25{suffix}.{d:%Y%m%d}.227.grib2"
    # Try the expected version first, then the others (transition dates overlap).
    ordered = [p for start, p in VERSIONS if d >= start][:1]
    ordered += [p for _, p in VERSIONS if p not in ordered]
    return [f"{BUCKET}/{p}/CS/{d:%Y%m%d}/{cycle:02d}/{name}" for p in ordered]


def extract_cycle(d: date, points: pd.DataFrame, cycle: int = 12,
                  bias_corrected: bool = True, overwrite: bool = False,
                  store: str = "west_us") -> pd.DataFrame | None:
    """Sample AQM hourly PM2.5 at points. Output: point_id, valid_utc, step, value."""
    tag = "bc" if bias_corrected else "raw"
    out_path = interim_dir(f"{store}/aqm_{tag}") / f"{d:%Y%m%d}_{cycle:02d}z.parquet"
    if out_path.exists() and not overwrite:
        return pd.read_parquet(out_path)
    blob = None
    for url in candidate_urls(d, cycle, bias_corrected):
        r = get(url, allow_404=True, timeout=300)
        if r is not None:
            blob = r.content
            break
    if blob is None:
        return None
    # AQM version changes can move cells without changing the grid dimensions.
    fields, grid = decode_messages(blob, with_grid=True)
    index = PointIndex(f"aqm{len(grid[0])}", grid[0], grid[1],
                       points["lat"].to_numpy(), points["lon"].to_numpy(), max_km=10.0)
    t0 = datetime(d.year, d.month, d.day, cycle)
    rows = []
    for f in fields:
        if f.end_step - f.start_step != 1:
            continue
        rows.append(pd.DataFrame({
            "point_id": points["point_id"].to_numpy(),
            "valid_utc": t0 + timedelta(hours=f.start_step),
            "step": np.int16(f.start_step),
            "value": index.sample(f.values),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out_path, index=False)
    return df


def load_cycles(days: list[date], cycle: int = 12, bias_corrected: bool = True,
                store: str = "west_us") -> pd.DataFrame:
    tag = "bc" if bias_corrected else "raw"
    parts = []
    for d in days:
        p = interim_dir(f"{store}/aqm_{tag}") / f"{d:%Y%m%d}_{cycle:02d}z.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["issue_date"] = d
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
