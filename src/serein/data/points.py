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


def nearest_site_offset(lat, lon, sites: pd.DataFrame | None) -> np.ndarray:
    """UTC offset of the nearest EPA site (exact, from AirNow); longitude guess if none.

    Longitude alone is wrong near time-zone borders (e.g. Boise is UTC-7 at -116.2).
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if sites is None or not len(sites) or "gmt_offset" not in sites:
        return standard_offset(lon)
    s = sites.dropna(subset=["lat", "lon", "gmt_offset"])
    if not len(s):
        return standard_offset(lon)
    slat, slon = np.radians(s["lat"].to_numpy(float)), np.radians(s["lon"].to_numpy(float))
    off = s["gmt_offset"].to_numpy(float)
    out = np.empty(len(lat), dtype=int)
    for i0 in range(0, len(lat), 2000):  # chunked to keep memory small
        a, b = np.radians(lat[i0:i0 + 2000])[:, None], np.radians(lon[i0:i0 + 2000])[:, None]
        d = np.sin((slat - a) / 2) ** 2 + np.cos(a) * np.cos(slat) * np.sin((slon - b) / 2) ** 2
        out[i0:i0 + 2000] = np.round(off[np.argmin(d, axis=1)]).astype(int)
    return out


def grid_points(region: Region, step: float, sites: pd.DataFrame | None = None) -> pd.DataFrame:
    if not np.isfinite(step) or step <= 0:
        raise ValueError("grid step must be finite and positive")
    lats = np.arange(region.lat_min, region.lat_max + 1e-9, step)
    lons = np.arange(region.lon_min, region.lon_max + 1e-9, step)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    precision = max(3, int(np.ceil(-np.log10(step))))
    df = pd.DataFrame({"lat": la.ravel().round(max(4, precision)),
                       "lon": lo.ravel().round(max(4, precision))})
    df["point_id"] = [f"G{a:+0{precision + 5}.{precision}f}{b:+0{precision + 6}.{precision}f}"
                      for a, b in zip(df["lat"], df["lon"])]
    df["kind"] = "grid"
    df["gmt_offset"] = nearest_site_offset(df["lat"], df["lon"], sites)
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
        p["gmt_offset"] = nearest_site_offset(p["lat"], p["lon"], s)
        parts.append(p[["point_id", "lat", "lon", "kind", "gmt_offset"]])
    if grid_step is not None:
        parts.append(grid_points(region, grid_step, s))
    pts = pd.concat(parts, ignore_index=True).drop_duplicates()
    if pts["point_id"].duplicated().any():
        raise ValueError("conflicting rows for the same point_id in points registry")
    return pts.reset_index(drop=True)


def save_points(pts: pd.DataFrame, name: str) -> None:
    pts.to_parquet(interim_dir(name) / PATH_NAME, index=False)


def load_points(name: str) -> pd.DataFrame:
    p = interim_dir(name) / PATH_NAME
    if not p.exists():
        raise FileNotFoundError(f"No points registry at {p}. Run `serein build-points` first.")
    return pd.read_parquet(p)
