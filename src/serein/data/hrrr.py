"""NOAA HRRR-Smoke and HRRR weather, sampled at points.

We only download the bytes we need. Every HRRR GRIB2 file on AWS has a
``.idx`` sidecar that lists where each field starts, so we can request just
the near-surface smoke field (``MASSDEN:8 m above ground``, ~0.8 MB) instead of
the whole 140 MB file. Example .idx line::

    76:50783879:d=2024090100:MASSDEN:8 m above ground:6 hour fcst:

Unit gotcha: MASSDEN was labeled µg/m³ before 2021-12-21 and kg/m³ after. We
decide by magnitude: a CONUS smoke field whose max is < 1 must be kg/m³.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from serein.config import interim_dir
from serein.data.grib import PointIndex, decode_messages
from serein.http import get, get_range

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
SMOKE = "MASSDEN:8 m above ground"
DEFAULT_MET = ["UGRD:10 m above ground", "VGRD:10 m above ground", "HPBL:surface",
               "TMP:2 m above ground", "RH:2 m above ground",
               "COLMD:entire atmosphere (considered as a single layer)"]
SHORT = {"MASSDEN:8 m above ground": "smoke", "UGRD:10 m above ground": "u10",
         "VGRD:10 m above ground": "v10", "HPBL:surface": "pbl",
         "TMP:2 m above ground": "t2m", "RH:2 m above ground": "rh2m",
         "COLMD:entire atmosphere (considered as a single layer)": "colmd"}
COLMD = "COLMD:entire atmosphere (considered as a single layer)"


def file_url(d: date, cycle: int, fhr: int) -> str:
    return f"{BUCKET}/hrrr.{d:%Y%m%d}/conus/hrrr.t{cycle:02d}z.wrfsfcf{fhr:02d}.grib2"


def parse_idx(text: str) -> list[tuple[int, int | None, str]]:
    """Return [(start, end_inclusive_or_None, 'VAR:level'), ...]."""
    rows = []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        parts = ln.split(":")
        start = int(parts[1])
        end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else None
        rows.append((start, end, f"{parts[3]}:{parts[4]}"))
    return rows


def colmd_to_mgm2(values: np.ndarray) -> np.ndarray:
    """Column smoke: kg/m2 -> mg/m2 (same magnitude rule as MASSDEN)."""
    v = values.astype(np.float32)
    finite = v[np.isfinite(v)]
    if finite.size and np.nanmax(finite) < 1.0:
        v = v * 1e6
    return v


def smoke_to_ugm3(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.float32)
    finite = v[np.isfinite(v)]
    if finite.size and np.nanmax(finite) < 1.0:
        v = v * 1e9  # kg/m³ -> µg/m³
    return v


def _fetch_fields(d: date, cycle: int, fhr: int, wanted: list[str]):
    url = file_url(d, cycle, fhr)
    r = get(url + ".idx", allow_404=True)
    if r is None:
        return None
    entries = parse_idx(r.text)
    blobs = []
    for start, end, name in entries:
        if name in wanted:
            blobs.append((name, get_range(url, start, end)))
    return blobs


def extract_cycle(d: date, points: pd.DataFrame, cycle: int = 12, max_fhr: int = 48,
                  smoke_every: int = 1, met_every: int = 3,
                  met_vars: list[str] | None = None, overwrite: bool = False,
                  workers: int = 12, store: str = "west_us") -> pd.DataFrame | None:
    """Sample HRRR smoke + weather at points for one cycle. Cached as parquet.

    Output columns: point_id, valid_utc, fhr, var, value
    """
    met_vars = met_vars or DEFAULT_MET
    # Cached per study ("store"): different studies sample different points.
    out_path = interim_dir(f"{store}/hrrr") / f"{d:%Y%m%d}_{cycle:02d}z.parquet"
    if out_path.exists() and not overwrite:
        return pd.read_parquet(out_path)

    jobs = []
    for fhr in range(max_fhr + 1):
        wanted = []
        if fhr % smoke_every == 0:
            wanted.append(SMOKE)
        if fhr % met_every == 0:
            wanted += met_vars
        if wanted:
            jobs.append((fhr, wanted))

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda j: (j[0], _fetch_fields(d, cycle, j[0], j[1])), jobs))

    if all(b is None for _, b in results):
        return None

    index = None
    rows = []
    t0 = datetime(d.year, d.month, d.day, cycle)
    for fhr, blobs in results:
        if not blobs:
            continue
        for name, blob in blobs:
            fields, grid = decode_messages(blob, with_grid=index is None)
            if not fields:
                continue
            if index is None:
                index = PointIndex("hrrr", grid[0], grid[1],
                                   points["lat"].to_numpy(), points["lon"].to_numpy(),
                                   max_km=5.0)
            for f in fields:
                if name == SMOKE:
                    vals = smoke_to_ugm3(f.values)
                elif name == COLMD:
                    vals = colmd_to_mgm2(f.values)
                else:
                    vals = f.values
                rows.append(pd.DataFrame({
                    "point_id": points["point_id"].to_numpy(),
                    "valid_utc": t0 + timedelta(hours=fhr),
                    "fhr": np.int16(fhr),
                    "var": SHORT.get(name, name),
                    "value": index.sample(vals),
                }))
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df["var"] = df["var"].astype("category")
    df.to_parquet(out_path, index=False)
    return df


def load_cycles(days: list[date], cycle: int = 12, store: str = "west_us") -> pd.DataFrame:
    parts = []
    for d in days:
        p = interim_dir(f"{store}/hrrr") / f"{d:%Y%m%d}_{cycle:02d}z.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["issue_date"] = d
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
