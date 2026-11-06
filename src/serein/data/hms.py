"""NOAA Hazard Mapping System (HMS) smoke plumes.

Trained analysts draw smoke polygons from satellite imagery every day and tag
each with a density: Light (~5 µg/m³), Medium (~16), Heavy (~27) column-smoke
classes. These describe smoke *somewhere in the column*, not at the ground, which
is exactly why we need sensors.

Daily shapefiles (free, no key):
  https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/YYYY/MM/hms_smokeYYYYMMDD.zip
"""

from __future__ import annotations

from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from serein.config import raw_dir
from serein.http import get

URL = ("https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/"
       "{d:%Y}/{d:%m}/hms_smoke{d:%Y%m%d}.zip")
DENSITY = {"light": 1, "medium": 2, "heavy": 3}


def download_day(d: date, overwrite: bool = False):
    path = raw_dir("hms") / f"hms_smoke{d:%Y%m%d}.zip"
    if path.exists() and not overwrite:
        return path
    r = get(URL.format(d=d), allow_404=True)
    if r is None:
        return None
    try:
        path.write_bytes(r.content)
    finally:
        r.close()
    return path


def read_day(d: date) -> gpd.GeoDataFrame | None:
    path = raw_dir("hms") / f"hms_smoke{d:%Y%m%d}.zip"
    if not path.exists():
        return None
    try:
        g = gpd.read_file(f"zip://{path}")
    except Exception:  # noqa: BLE001 - unreadable file = missing day (None), not "no smoke"
        return None
    if g.empty:
        return g
    col = next((c for c in g.columns if c.lower() == "density"), None)
    g["density_level"] = (
        g[col].astype(str).str.strip().str.lower().map(DENSITY).fillna(1).astype(int)
        if col else 1
    )
    if g.crs is None:
        g = g.set_crs(4326)
    return g.to_crs(4326)


def density_at_points(d: date, points: pd.DataFrame) -> np.ndarray:
    """Max density (0 none .. 3 heavy); NaN if the day's smoke map is unavailable."""
    g = read_day(d)
    if g is None:
        return np.full(len(points), np.nan)
    out = np.zeros(len(points), dtype=float)
    if g.empty:
        return out
    pts = gpd.GeoDataFrame(
        {"i": np.arange(len(points))},
        geometry=[Point(xy) for xy in zip(points["lon"], points["lat"])], crs=4326,
    )
    j = gpd.sjoin(pts, g[["density_level", "geometry"]], predicate="within", how="inner")
    if len(j):
        m = j.groupby("i")["density_level"].max()
        out[m.index.to_numpy()] = m.to_numpy()
    return out


# --------------------------------------------------------------------------- fires
FIRE_URL = ("https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Fire_Points/Text/"
            "{d:%Y}/{d:%m}/hms_fire{d:%Y%m%d}.txt")


def download_fire_day(d: date, overwrite: bool = False):
    """HMS fire detections (no key needed): analyst-checked VIIRS/MODIS/GOES points
    with fire radiative power (FRP, MW). Saved as parquet with the same column names
    as NASA FIRMS (latitude, longitude, frp, time_utc), so fire features work with either.
    """
    import io

    path = raw_dir("hms_fire") / f"{d:%Y%m%d}.parquet"
    if path.exists() and not overwrite:
        return path
    r = get(FIRE_URL.format(d=d), allow_404=True)
    if r is None:
        return None
    try:
        text = r.text
    finally:
        r.close()
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    t = pd.to_datetime(df["yearday"].astype(int).astype(str), format="%Y%j")
    hhmm = df["time"].astype(int)
    out = pd.DataFrame({
        "latitude": df["lat"].astype(float), "longitude": df["lon"].astype(float),
        "frp": pd.to_numeric(df["frp"], errors="coerce").fillna(0.0).clip(lower=0),
        "time_utc": t + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m"),
        "satellite": df["satellite"].astype(str).str.strip(),
        "method": df["method"].astype(str).str.strip(),
    })
    out.to_parquet(path, index=False)
    return path


def load_fires(days) -> pd.DataFrame:
    """HMS fire points for the given days (plus the day before each, for 24 h windows)."""
    want = sorted({d + pd.Timedelta(days=k).to_pytimedelta() for d in days for k in (-1, 0)})
    parts = [pd.read_parquet(p) for d in want
             if (p := raw_dir("hms_fire") / f"{d:%Y%m%d}.parquet").exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
