"""Auditable, offline PurpleAir quality control (README task A-12).

This supplements the existing A/B check and EPA correction. Stuck-reading and
neighbor thresholds are configurable screening heuristics, not validated sensor
fault classifiers. Original rows are retained with reasons; only accepted rows
receive ``pm25``. See ``docs/PURPLEAIR_QC.md`` for the input schema and limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from serein.data.firms import haversine_km
from serein.data.purpleair import epa_correction, qc_ab

REJECTION_FLAGS = (
    "invalid_sensor", "invalid_time", "duplicate", "indoor", "unknown_location",
    "invalid_channels", "ab_disagreement", "invalid_humidity", "missing_coordinates",
    "stuck", "neighbor_outlier",
)


@dataclass(frozen=True)
class QCConfig:
    ab_abs_tol: float = 5.0
    ab_rel_tol: float = 0.7
    expected_minutes: int = 60
    stuck_hours: float = 24.0
    stuck_tolerance: float = 0.01
    neighbor_radius_km: float = 25.0
    min_neighbors: int = 3
    neighbor_abs_tol: float = 30.0
    neighbor_rel_tol: float = 2.0

    def __post_init__(self):
        for name in ("expected_minutes", "min_neighbors"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("ab_abs_tol", "ab_rel_tol", "stuck_hours", "stuck_tolerance",
                     "neighbor_radius_km", "neighbor_abs_tol", "neighbor_rel_tol"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.stuck_hours <= 0 or self.neighbor_radius_km <= 0:
            raise ValueError("stuck_hours and neighbor_radius_km must be positive")


def _column(frame: pd.DataFrame, name: str, alias: str | None = None) -> pd.Series:
    if name in frame:
        return frame[name].reset_index(drop=True)
    if alias is not None and alias in frame:
        return frame[alias].reset_index(drop=True)
    return pd.Series(np.nan, index=range(len(frame)), dtype=float)


def _sensor_ids(values: pd.Series) -> pd.Series:
    ids = pd.to_numeric(values, errors="coerce").astype(float)
    invalid = (~np.isfinite(ids) | (ids < 1) | (ids % 1 != 0)
               | values.map(lambda v: isinstance(v, (bool, np.bool_))))
    return ids.mask(invalid)


def _location(value, *, indoor_field: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value) if indoor_field else np.nan
    if isinstance(value, str):
        value = value.strip().lower()
        names = {"indoor": 1, "outdoor": 0, "0": 0, "1": 1}
        if indoor_field:
            names.update({"true": 1, "false": 0, "yes": 1, "no": 0})
        return float(names.get(value, np.nan))
    if isinstance(value, Real) and np.isfinite(value) and value in (0, 1):
        return float(value)
    return np.nan


def _attach_metadata(history: pd.DataFrame, sensors: pd.DataFrame | None) -> pd.DataFrame:
    out = history.copy(deep=True)
    if sensors is None:
        return out
    if "sensor_index" not in sensors:
        raise ValueError("sensor metadata must contain sensor_index")
    meta_ids = _sensor_ids(_column(sensors, "sensor_index"))
    if meta_ids.isna().any() or meta_ids.duplicated().any():
        raise ValueError("sensor metadata must have unique positive integer sensor_index values")
    ids = _sensor_ids(_column(out, "sensor_index"))
    for name, alias in (("lat", "latitude"), ("lon", "longitude"),
                        ("location_type", None), ("indoor", None)):
        if name not in sensors and (alias is None or alias not in sensors):
            continue
        lookup = pd.Series(_column(sensors, name, alias).to_numpy(), index=meta_ids)
        existing = _column(out, name, alias)
        # Keep source values; metadata fills unavailable fields only.
        out[name] = existing.where(existing.notna(), ids.map(lookup)).to_numpy()
    return out


def _stuck_readings(ids, times, a, b, eligible, cfg: QCConfig) -> np.ndarray:
    stuck = np.zeros(len(ids), dtype=bool)
    step = pd.Timedelta(minutes=cfg.expected_minutes)
    minimum = max(2, int(np.ceil(cfg.stuck_hours * 60 / cfg.expected_minutes)))
    rows = pd.DataFrame({"sensor": ids, "time": times, "a": a, "b": b,
                         "eligible": eligible})
    for _, group in rows[ids.notna() & times.notna()].groupby("sensor", sort=False):
        group = group.sort_values("time", kind="stable")
        positions, run = group.index.to_numpy(), []
        for i in positions:
            consecutive = (run and times.iloc[i] - times.iloc[run[-1]] == step
                           and abs(a[i] - a[run[0]]) <= cfg.stuck_tolerance
                           and abs(b[i] - b[run[0]]) <= cfg.stuck_tolerance)
            if not eligible[i] or not consecutive:
                if len(run) >= minimum:
                    stuck[run[minimum - 1:]] = True
                run = []
            if eligible[i]:
                run.append(i)
        if len(run) >= minimum:
            stuck[run[minimum - 1:]] = True
    return stuck


def _neighbor_flags(ids, times, lat, lon, pm, eligible, cfg: QCConfig):
    flagged = np.zeros(len(ids), dtype=bool)
    evaluated = np.zeros(len(ids), dtype=bool)
    rows = pd.DataFrame({"sensor": ids, "time": times, "hour": times.dt.floor("h"),
                         "lat": lat, "lon": lon, "pm": pm})
    # Each sensor contributes once per hour, regardless of sampling frequency.
    for _, group in rows[eligible].groupby("hour", sort=False):
        for cutoff, current in group.groupby("time", sort=True):
            # Later subhourly observations must not change an earlier row's QC.
            peers = group[group["time"] <= cutoff].groupby("sensor", sort=False).agg(
                lat=("lat", "first"), lon=("lon", "first"), pm=("pm", "mean"))
            for sensor in current["sensor"].unique():
                own = peers.loc[sensor]
                distance = haversine_km(own.lat, own.lon, peers["lat"].to_numpy(),
                                        peers["lon"].to_numpy())
                nearby = (distance <= cfg.neighbor_radius_km) & (peers.index != sensor)
                if nearby.sum() < cfg.min_neighbors:
                    continue
                median = float(np.median(peers.loc[nearby, "pm"]))
                positions = current.index[current["sensor"] == sensor].to_numpy()
                evaluated[positions] = True
                difference = np.abs(pm[positions] - median)
                flagged[positions] = ((difference > cfg.neighbor_abs_tol)
                                      & (difference > cfg.neighbor_rel_tol * max(median, 1.0)))
    return flagged, evaluated


def quality_control(history: pd.DataFrame, sensors: pd.DataFrame | None = None,
                    config: QCConfig | None = None) -> pd.DataFrame:
    """Keep every input row and append QC flags plus corrected PM for accepted rows.

    Required values (absent fields are flagged): sensor_index, time_utc (or epoch
    seconds time_stamp), pm2.5_cf_1_a/b, humidity, lat/lon, and location_type
    (0 outdoor / 1 indoor) or indoor (boolean). Metadata may supply the latter
    three fields. Naive time_utc timestamps are interpreted as UTC.
    """
    if not isinstance(history, pd.DataFrame):
        raise TypeError("history must be a pandas DataFrame")
    if not history.columns.is_unique:
        raise ValueError("history column names must be unique")
    if sensors is not None and not isinstance(sensors, pd.DataFrame):
        raise TypeError("sensors must be a pandas DataFrame")
    if sensors is not None and not sensors.columns.is_unique:
        raise ValueError("sensor metadata column names must be unique")
    if config is not None and not isinstance(config, QCConfig):
        raise TypeError("config must be QCConfig")
    if any(str(c).startswith("qc_") or c in ("pm25", "pm25_cf1_mean", "time_utc_normalized")
           for c in history.columns):
        raise ValueError("input must be raw history without QC output columns")
    cfg = config or QCConfig()
    original_index = history.index
    out = _attach_metadata(history.reset_index(drop=True), sensors)
    ids = _sensor_ids(_column(out, "sensor_index"))
    if "time_utc" in out:
        source_time = out["time_utc"]
        times = pd.to_datetime(source_time, utc=True, format="mixed", errors="coerce")
        numeric_time = source_time.map(lambda v: isinstance(v, Real))
        times = times.mask(numeric_time)
    else:
        stamp = _column(out, "time_stamp")
        stamp = stamp.mask(stamp.map(lambda v: isinstance(v, (bool, np.bool_))))
        times = pd.to_datetime(pd.to_numeric(stamp, errors="coerce"),
                               unit="s", utc=True, errors="coerce")
    a, b, humidity = (pd.to_numeric(_column(out, c), errors="coerce").to_numpy(float)
                      for c in ("pm2.5_cf_1_a", "pm2.5_cf_1_b", "humidity"))
    lat, lon = (pd.to_numeric(_column(out, name, alias), errors="coerce").to_numpy(float)
                for name, alias in (("lat", "latitude"), ("lon", "longitude")))
    loc = _column(out, "location_type").map(_location)
    indoor = _column(out, "indoor").map(lambda v: _location(v, indoor_field=True))
    conflict = loc.notna() & indoor.notna() & (loc != indoor)
    # Malformed raw readings must produce flags, including values that overflow
    # the correction, rather than warnings or an infinite corrected measurement.
    with np.errstate(invalid="ignore", over="ignore"):
        mean, ab_keep = qc_ab(a, b, cfg.ab_abs_tol, cfg.ab_rel_tol)
    valid_channels = (np.isfinite(a) & np.isfinite(b) & np.isfinite(mean)
                      & (a >= 0) & (b >= 0))
    flags = {
        "invalid_sensor": ids.isna().to_numpy(),
        "invalid_time": times.isna().to_numpy(),
        "duplicate": pd.DataFrame({"sensor": ids, "time": times}).duplicated(
            ["sensor", "time"], keep=False).to_numpy(),
        "indoor": ((loc == 1) | (indoor == 1)).to_numpy(),
        "unknown_location": ((loc.isna() & indoor.isna()) | conflict).to_numpy(),
        "invalid_channels": ~valid_channels,
        "ab_disagreement": valid_channels & ~ab_keep,
        "invalid_humidity": ~np.isfinite(humidity) | (humidity < 0) | (humidity > 100),
        "missing_coordinates": (~np.isfinite(lat) | ~np.isfinite(lon)
                                 | (np.abs(lat) > 90) | (np.abs(lon) > 180)),
    }
    eligible = ~np.logical_or.reduce(list(flags.values()))
    flags["stuck"] = _stuck_readings(ids, times, a, b, eligible, cfg)
    eligible &= ~flags["stuck"]
    corrected = np.full(len(out), np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        corrected[eligible] = epa_correction(mean[eligible], humidity[eligible])
    invalid_correction = eligible & ~np.isfinite(corrected)
    flags["invalid_channels"] |= invalid_correction
    eligible &= ~invalid_correction
    corrected[~eligible] = np.nan
    flags["neighbor_outlier"], evaluated = _neighbor_flags(
        ids, times, lat, lon, corrected, eligible, cfg)
    keep = eligible & ~flags["neighbor_outlier"]
    for name in REJECTION_FLAGS:
        out[f"qc_{name}"] = flags[name]
    out["qc_neighbor_evaluated"] = evaluated
    out["qc_flags"] = [";".join(name for name in REJECTION_FLAGS if flags[name][i])
                       for i in range(len(out))]
    out["qc_keep"] = keep
    out["time_utc_normalized"] = times.dt.tz_localize(None)
    out["pm25_cf1_mean"] = mean
    out["pm25"] = np.where(keep, corrected, np.nan)
    out.index = original_index
    return out


def summarize_qc(audit: pd.DataFrame) -> dict:
    """Counts can overlap: one rejected row may fail more than one rule."""
    return {"rows": len(audit), "kept": int(audit["qc_keep"].sum()),
            "rejected": int((~audit["qc_keep"]).sum()),
            "neighbor_evaluated": int(audit["qc_neighbor_evaluated"].sum()),
            "flags": {name: int(audit[f"qc_{name}"].sum()) for name in REJECTION_FLAGS}}


def accepted_history(audit: pd.DataFrame) -> pd.DataFrame:
    """Return the three-column format consumed by future sensor integration."""
    if audit.empty:
        return pd.DataFrame(columns=["time_utc", "sensor_index", "pm25"])
    keep = audit["qc_keep"]
    return pd.DataFrame({"time_utc": audit.loc[keep, "time_utc_normalized"].to_numpy(),
                         "sensor_index": audit.loc[keep, "sensor_index"].to_numpy(),
                         "pm25": audit.loc[keep, "pm25"].to_numpy()})
