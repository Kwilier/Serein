"""The registry of places we forecast for ("points").

Points are EPA monitors, PurpleAir sensors, and a regular grid of "virtual
sensors" used to draw neighborhood maps. Model inputs (HRRR, AQM, HMS, fires)
are sampled at every point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from serein.config import Region, interim_dir

PATH_NAME = "points.parquet"


def standard_offset(lon) -> np.ndarray:
    """Rough standard-time UTC offset from longitude (-120 -> -8, -75 -> -5).

    Good enough for map cells; real sensors use AirNow's exact offset.
    """
    return np.clip(np.round(np.asarray(lon, float) / 15.0), -10, -4).astype(int)


def grid_points(region: Region, step: float) -> pd.DataFrame:
    lats = np.arange(region.lat_min, region.lat_max + 1e-9, step)
    lons = np.arange(region.lon_min, region.lon_max + 1e-9, step)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({"lat": la.ravel().round(4), "lon": lo.ravel().round(4)})
    df["point_id"] = [f"G{a:+08.3f}{b:+09.3f}" for a, b in zip(df["lat"], df["lon"])]
    df["kind"] = "grid"
    df["gmt_offset"] = standard_offset(df["lon"])
    return df


def build_points(sites: pd.DataFrame, region: Region, grid_step: float | None,
                 purpleair: pd.DataFrame | None = None) -> pd.DataFrame:
    s = sites[region.contains(sites["lat"], sites["lon"])].copy()
    s = s.rename(columns={"site_id": "point_id"})
    s["kind"] = "epa"
    if "reporting_area" not in s:
        s["reporting_area"] = ""
    parts = [s[["point_id", "lat", "lon", "kind", "gmt_offset", "reporting_area"]]]
    if purpleair is not None and len(purpleair):
        p = purpleair[region.contains(purpleair["lat"], purpleair["lon"])].copy()
        p["point_id"] = "PA" + p["sensor_index"].astype(str)
        p["kind"] = "purpleair"
        p["gmt_offset"] = standard_offset(p["lon"])
        parts.append(p[["point_id", "lat", "lon", "kind", "gmt_offset"]])
    if grid_step:
        parts.append(grid_points(region, grid_step))
    pts = pd.concat(parts, ignore_index=True).drop_duplicates("point_id")
    return pts.reset_index(drop=True)


def save_points(pts: pd.DataFrame, name: str) -> None:
    pts.to_parquet(interim_dir(name) / PATH_NAME, index=False)


def load_points(name: str) -> pd.DataFrame:
    p = interim_dir(name) / PATH_NAME
    if not p.exists():
        raise FileNotFoundError(f"No points registry at {p}. Run `serein build-points` first.")
    return pd.read_parquet(p)
