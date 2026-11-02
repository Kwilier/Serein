"""PurpleAir low-cost sensors: client, cost estimator, QC, and EPA correction.

API docs: https://api.purpleair.com/  (key in .env as PURPLEAIR_API_KEY=...)

PurpleAir charges "points" per call: a small per-call base plus a per-row cost for
each field. ALWAYS run ``estimate_history_points`` and get the team's OK before a
big download. Real prices live on the PurpleAir Developer Dashboard (Billing ->
Pricing); the defaults below are placeholders to update from that page.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from serein.config import Region, load_env, raw_dir
from serein.http import get

API = "https://api.purpleair.com/v1"
HISTORY_FIELDS = ["pm2.5_cf_1_a", "pm2.5_cf_1_b", "humidity"]


def _headers() -> dict:
    load_env()
    k = os.environ.get("PURPLEAIR_API_KEY")
    if not k:
        raise RuntimeError("PURPLEAIR_API_KEY is not set (put it in .env)")
    return {"X-API-Key": k}


# --------------------------------------------------------------------------- cost
@dataclass
class PointPrices:
    """UPDATE these from the Developer Dashboard before trusting the estimate."""
    base_per_call: float = 2.0
    per_field_row: float = 1.0
    max_days_per_call_hourly: int = 14  # history call span limit for 60-min averages


def estimate_history_points(n_sensors: int, days: int, average_minutes: int = 60,
                            fields: list[str] = HISTORY_FIELDS,
                            prices: PointPrices = PointPrices()) -> dict:
    rows_per_sensor = days * 24 * 60 / average_minutes
    calls = n_sensors * int(np.ceil(days / prices.max_days_per_call_hourly))
    points = calls * prices.base_per_call + n_sensors * rows_per_sensor * len(fields) * prices.per_field_row
    return {"sensors": n_sensors, "days": days, "calls": calls,
            "rows": int(n_sensors * rows_per_sensor), "points": int(points)}


# ------------------------------------------------------------------------ client
def list_outdoor_sensors(region: Region) -> pd.DataFrame:
    params = {
        "fields": "name,latitude,longitude,location_type,date_created,last_seen",
        "location_type": 0,  # outdoor only
        "nwlng": region.lon_min, "nwlat": region.lat_max,
        "selat": region.lat_min, "selng": region.lon_max,
    }
    r = get(f"{API}/sensors", headers=_headers(), params=params)
    js = r.json()
    df = pd.DataFrame(js["data"], columns=js["fields"])
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df.to_parquet(raw_dir("purpleair") / "sensors.parquet", index=False)
    return df


def download_history(sensor_index: int, start: datetime, end: datetime,
                     average_minutes: int = 60, fields: list[str] = HISTORY_FIELDS,
                     max_days: int = 14, pause_s: float = 1.0) -> pd.DataFrame:
    path = raw_dir("purpleair") / "history" / f"{sensor_index}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_parquet(path)
    parts, t = [], start
    while t < end:
        t1 = min(t + timedelta(days=max_days), end)
        params = {
            "start_timestamp": int(t.replace(tzinfo=timezone.utc).timestamp()),
            "end_timestamp": int(t1.replace(tzinfo=timezone.utc).timestamp()),
            "average": average_minutes, "fields": ",".join(fields),
        }
        r = get(f"{API}/sensors/{sensor_index}/history", headers=_headers(), params=params)
        js = r.json()
        if js.get("data"):
            parts.append(pd.DataFrame(js["data"], columns=js["fields"]))
        t = t1
        time.sleep(pause_s)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["time_utc"] = pd.to_datetime(df["time_stamp"], unit="s")
    df["sensor_index"] = sensor_index
    df.to_parquet(path, index=False)
    return df


# --------------------------------------------------------------------- QC + fix
def qc_ab(a, b, abs_tol: float = 5.0, rel_tol: float = 0.7):
    """Keep hours where channels A and B agree (within 5 µg/m³ OR within 70%).

    Returns (mean of A and B, keep mask). Mirrors the A/B agreement check used
    by EPA for the AirNow Fire and Smoke Map.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mean = (a + b) / 2
    diff = np.abs(a - b)
    rel = np.where(mean > 0, diff / np.maximum(mean, 1e-9), 0)
    keep = np.isfinite(a) & np.isfinite(b) & ((diff <= abs_tol) | (rel <= rel_tol))
    return mean, keep


def epa_correction(pa_cf1, rh):
    """EPA U.S.-wide PurpleAir correction with the extension for heavy smoke.

    Barkjohn et al. (2021, AMT 14:4617) for typical levels, extended for wildfire
    smoke by Barkjohn et al. (2022, Sensors 22:9669). Input: mean cf_1 PM2.5 of
    channels A and B (µg/m³) and the sensor's relative humidity (%).
    TODO(A-12): double-check coefficients against the papers before the final test.
    """
    x = np.asarray(pa_cf1, dtype=float)
    rh = np.asarray(rh, dtype=float)
    out = np.full(x.shape, np.nan)

    m = x < 30
    out[m] = 0.524 * x[m] - 0.0862 * rh[m] + 5.75

    m = (x >= 30) & (x < 50)
    w = x[m] / 20 - 3 / 2
    out[m] = (0.786 * w + 0.524 * (1 - w)) * x[m] - 0.0862 * rh[m] + 5.75

    m = (x >= 50) & (x < 210)
    out[m] = 0.786 * x[m] - 0.0862 * rh[m] + 5.75

    m = (x >= 210) & (x < 260)
    w = x[m] / 50 - 21 / 5
    out[m] = ((0.69 * w + 0.786 * (1 - w)) * x[m] - 0.0862 * rh[m] * (1 - w)
              + 2.966 * w + 5.75 * (1 - w) + 8.84e-4 * x[m] ** 2 * w)

    m = x >= 260
    out[m] = 2.966 + 0.69 * x[m] + 8.84e-4 * x[m] ** 2
    return np.clip(out, 0, None)


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """A/B QC + EPA correction -> columns time_utc, sensor_index, pm25."""
    mean, keep = qc_ab(df["pm2.5_cf_1_a"], df["pm2.5_cf_1_b"])
    rh = df["humidity"].fillna(df["humidity"].median()).to_numpy()
    pm = epa_correction(mean, rh)
    out = pd.DataFrame({"time_utc": df["time_utc"], "sensor_index": df["sensor_index"], "pm25": pm})
    return out[keep].reset_index(drop=True)
