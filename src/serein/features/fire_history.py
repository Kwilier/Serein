"""Optional, availability-aware fire history for development experiments.

Coverage declarations describe acquisition intervals in a fully retrieved source
archive, not satellite visibility or proof that no fire was burning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from serein.data.firms import haversine_km

HISTORY_COLUMNS = (
    "fire_recent_complete", "fire_recent_count", "fire_recent_frp", "fire_latest_age_hours",
    "fire_previous_complete", "fire_previous_count", "fire_previous_frp",
    "fire_frp_log_change", "fire_count_change",
)


def _times(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], utc=True, format="mixed")
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing timestamps")
    return frame


def _coordinates(frame: pd.DataFrame, lat: str, lon: str) -> None:
    for column, limit in ((lat, 90), (lon, 180)):
        if np.iscomplexobj(frame[column].to_numpy()):
            raise ValueError(f"{column} must contain real coordinates")
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        if not (np.isfinite(values) & (np.abs(values) <= limit)).all():
            raise ValueError(f"{column} must contain finite coordinates within +/-{limit}")
        frame[column] = values


def _covered(intervals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    cursor = start
    for row in intervals.sort_values("start_utc").itertuples():
        if row.end_utc <= cursor:
            continue
        if row.start_utc > cursor:
            return False
        cursor = max(cursor, row.end_utc)
        if cursor >= end:
            return True
    return False


def _coverage_at_points(intervals, points, start, end, radius_km):
    intervals = intervals[intervals.end_utc.gt(start) & intervals.start_utc.lt(end)]
    complete = np.zeros(len(points), dtype=bool)
    if intervals.empty or points.empty:
        return complete
    angle = min(radius_km / 6371., np.pi)
    latitude = points.lat.to_numpy(dtype=float)
    longitude = points.lon.to_numpy(dtype=float)
    lat_low = np.maximum(-90., latitude - np.degrees(angle))
    lat_high = np.minimum(90., latitude + np.degrees(angle))
    crosses_pole = np.abs(np.radians(latitude)) + angle >= np.pi / 2
    longitude_span = np.degrees(np.arcsin(np.clip(
        np.sin(angle) / np.maximum(np.cos(np.radians(latitude)), 1e-15), -1, 1)))
    longitude_span = np.where(crosses_pole, 180., longitude_span)
    lon_low, lon_high = longitude - longitude_span, longitude + longitude_span
    wraps = (lon_low < -180) | (lon_high > 180) | crosses_pole
    lon_low, lon_high = np.where(wraps, -180., lon_low), np.where(wraps, 180., lon_high)
    matches = ((lat_low[:, None] >= intervals.lat_min.to_numpy()[None, :])
               & (lat_high[:, None] <= intervals.lat_max.to_numpy()[None, :])
               & (lon_low[:, None] >= intervals.lon_min.to_numpy()[None, :])
               & (lon_high[:, None] <= intervals.lon_max.to_numpy()[None, :]))
    patterns, groups = np.unique(matches, axis=0, return_inverse=True)
    for i, pattern in enumerate(patterns):
        if pattern.any() and _covered(intervals.iloc[np.flatnonzero(pattern)], start, end):
            complete[groups == i] = True
    return complete


def fire_history_features(
    fires: pd.DataFrame,
    coverage: pd.DataFrame,
    points: pd.DataFrame,
    issue_time,
    *,
    source: str,
    window_hours: float = 24,
    radius_km: float = 250,
    acquisition_lag_hours: float = 0,
) -> pd.DataFrame:
    """Compare adjacent acquisition windows using only records available at issue.

    Windows end acquisition_lag_hours before issue. Both are open on the left
    and closed on the right; the previous window ends where the recent one starts.
    Input times without a timezone are UTC. Exact duplicate detections are counted
    once. Conflicting FRP values for one source/location/time raise an error.
    Every input detection must have an explicit available_utc timestamp. No lag is
    inferred from acquisition time, file modification time, or processing product.
    """
    for name, value in (("window_hours", window_hours), ("radius_km", radius_km)):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (isinstance(acquisition_lag_hours, (bool, np.bool_))
            or not np.isfinite(acquisition_lag_hours) or acquisition_lag_hours < 0):
        raise ValueError("acquisition_lag_hours must be finite and nonnegative")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a nonempty source identifier")
    issue = pd.Timestamp(issue_time)
    if pd.isna(issue):
        raise ValueError("issue_time must be a timestamp")
    issue = issue.tz_localize("UTC") if issue.tzinfo is None else issue.tz_convert("UTC")
    delta = pd.Timedelta(hours=window_hours)
    if delta <= pd.Timedelta(0):
        raise ValueError("window_hours is below timestamp resolution")
    cutoff = issue - pd.Timedelta(hours=acquisition_lag_hours)
    points = points.copy()
    _coordinates(points, "lat", "lon")
    if points["point_id"].isna().any() or points["point_id"].duplicated().any():
        raise ValueError("point_id must be present and unique")
    for name, frame in (("fires", fires), ("coverage", coverage)):
        if frame["source"].isna().any():
            raise ValueError(f"{name} contains missing source identifiers")
    f = _times(fires[fires.source.eq(source)], ("time_utc", "available_utc"))
    c = _times(coverage[coverage.source.eq(source)],
               ("start_utc", "end_utc", "available_utc"))
    if (f.available_utc < f.time_utc).any():
        raise ValueError("detections cannot be available before acquisition")
    if (c.end_utc <= c.start_utc).any():
        raise ValueError("coverage end_utc must follow start_utc")
    if (c.available_utc < c.end_utc).any():
        raise ValueError("coverage cannot be available before its interval ends")
    for lat, lon in (("lat_min", "lon_min"), ("lat_max", "lon_max")):
        _coordinates(c, lat, lon)
    if (c.lat_min.ge(c.lat_max) | c.lon_min.ge(c.lon_max)).any():
        raise ValueError("coverage rectangles must have increasing latitude and longitude bounds")
    f = f[f.available_utc.le(issue) & f.time_utc.gt(cutoff - 2 * delta)
          & f.time_utc.le(cutoff)].copy()
    c = c[c.available_utc.le(issue)]
    _coordinates(f, "latitude", "longitude")
    if np.iscomplexobj(f.frp.to_numpy()):
        raise ValueError("frp must contain real values")
    frp = f.frp.to_numpy(dtype=float, na_value=np.nan)
    if (np.isinf(frp) | (frp < 0)).any():
        raise ValueError("frp must be nonnegative and finite or missing")
    f["frp"] = frp
    keys = ["latitude", "longitude", "time_utc"]
    if f.groupby(keys, dropna=False).frp.nunique(dropna=False).gt(1).any():
        raise ValueError("conflicting FRP for a source/location/acquisition")
    f = f.drop_duplicates(keys)
    out = points[["point_id"]].reset_index(drop=True).copy()
    out["issue_time"] = pd.Series(issue, index=out.index, dtype="datetime64[ns, UTC]")
    out["source"] = source
    out["window_hours"] = window_hours
    out["radius_km"] = radius_km
    out["window_end_utc"] = pd.Series(cutoff, index=out.index, dtype="datetime64[ns, UTC]")
    out["acquisition_lag_hours"] = acquisition_lag_hours
    for label, start, end in (("recent", cutoff - delta, cutoff),
                              ("previous", cutoff - 2 * delta, cutoff - delta)):
        complete = _coverage_at_points(c, points, start, end, radius_km)
        out[f"fire_{label}_complete"] = complete
        out[f"fire_{label}_count"] = np.nan
        out[f"fire_{label}_frp"] = np.nan
        if label == "recent":
            out["fire_latest_age_hours"] = np.nan
        if not complete.any():
            continue
        selected = f[f.time_utc.gt(start) & f.time_utc.le(end)]
        # Bound the point-by-detection matrix independently of archive size.
        step = max(1, 1_000_000 // max(len(selected), 1))
        ages = (issue - selected.time_utc).dt.total_seconds().to_numpy() / 3600
        power = selected.frp.to_numpy(dtype=float)
        positions = np.flatnonzero(complete)
        for offset in range(0, len(positions), step):
            index = positions[offset:offset + step]
            block = points.iloc[index]
            distance = haversine_km(block.lat.to_numpy()[:, None],
                                    block.lon.to_numpy()[:, None],
                                    selected.latitude.to_numpy()[None, :],
                                    selected.longitude.to_numpy()[None, :])
            near = distance <= radius_km
            out.loc[index, f"fire_{label}_count"] = near.sum(axis=1)
            out.loc[index, f"fire_{label}_frp"] = np.where(near, power, 0).sum(axis=1)
            if label == "recent" and len(selected):
                age = np.where(near, ages, np.inf).min(axis=1)
                out.loc[index, "fire_latest_age_hours"] = np.where(np.isfinite(age), age, np.nan)
    out["fire_frp_log_change"] = (np.log1p(out.fire_recent_frp)
                                   - np.log1p(out.fire_previous_frp))
    out["fire_count_change"] = out.fire_recent_count - out.fire_previous_count
    return out


def augment_fire_history(
    table: pd.DataFrame,
    fires: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    source: str,
    issue_hour_utc: float,
    window_hours: float = 24,
    radius_km: float = 250,
    acquisition_lag_hours: float = 0,
) -> pd.DataFrame:
    """Append numeric history features to a copy of an existing development table.

    No targets, row ordering, identifiers or original features are changed. Source
    and parameter metadata belong in an experiment record, not numeric model inputs.
    """
    if (isinstance(issue_hour_utc, (bool, np.bool_)) or not np.isfinite(issue_hour_utc)
            or not 0 <= issue_hour_utc < 24):
        raise ValueError("issue_hour_utc must be finite and in [0, 24)")
    labels = [pd.Timestamp(value) for value in table.issue_date]
    if any(pd.isna(value) or value != value.normalize() for value in labels):
        raise ValueError("issue_date must contain midnight calendar labels")
    dates = pd.Series([value.date() for value in labels])
    keys = pd.DataFrame({"issue_date": dates, "point_id": table.point_id.to_numpy()})
    if keys.point_id.isna().any() or keys.duplicated().any():
        raise ValueError("table requires unique, nonmissing issue_date/point_id pairs")
    if set(HISTORY_COLUMNS).intersection(table.columns):
        raise ValueError("table already contains fire-history columns")
    out = table.copy()
    parts = []
    for day in dates.unique():
        positions = np.flatnonzero(dates.eq(day))
        features = fire_history_features(
            fires, coverage, table.iloc[positions][["point_id", "lat", "lon"]],
            pd.Timestamp(day) + pd.Timedelta(hours=issue_hour_utc), source=source,
            window_hours=window_hours, radius_km=radius_km,
            acquisition_lag_hours=acquisition_lag_hours)
        features.index = positions
        parts.append(features[list(HISTORY_COLUMNS)])
    if not parts:
        raise ValueError("table must contain at least one issue date")
    features = pd.concat(parts).sort_index()
    for column in features:
        out[column] = features[column].to_numpy()
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fires", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--points", type=Path)
    mode.add_argument("--table", type=Path)
    parser.add_argument("--issue-time")
    parser.add_argument("--issue-hour-utc", type=float)
    parser.add_argument("--source", required=True)
    parser.add_argument("--window-hours", type=float, default=24)
    parser.add_argument("--radius-km", type=float, default=250)
    parser.add_argument("--acquisition-lag-hours", type=float, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.out.with_suffix(args.out.suffix + ".json")
    if args.out.exists() or manifest_path.exists():
        parser.error("output or manifest already exists; choose a new path")
    paths = {"fires": args.fires, "coverage": args.coverage,
             "table" if args.table else "points": args.table or args.points}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    fires, coverage = pd.read_parquet(args.fires), pd.read_parquet(args.coverage)
    options = {"source": args.source, "window_hours": args.window_hours,
               "radius_km": args.radius_km, "acquisition_lag_hours": args.acquisition_lag_hours}
    if args.table:
        if args.issue_hour_utc is None or args.issue_time is not None:
            parser.error("--table requires --issue-hour-utc and forbids --issue-time")
        result = augment_fire_history(pd.read_parquet(args.table), fires, coverage,
                                      issue_hour_utc=args.issue_hour_utc, **options)
    else:
        if args.issue_time is None or args.issue_hour_utc is not None:
            parser.error("--points requires --issue-time and forbids --issue-hour-utc")
        result = fire_history_features(fires, coverage, pd.read_parquet(args.points),
                                       args.issue_time, **options)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        result.to_parquet(stream, index=args.table is not None)
    manifest = {"source": args.source, "window_hours": args.window_hours,
                "radius_km": args.radius_km, "acquisition_lag_hours": args.acquisition_lag_hours,
                "issue_time": args.issue_time, "issue_hour_utc": args.issue_hour_utc,
                "rows": len(result), "inputs": {
                    name: {"path": str(path.resolve()), "sha256": hashes[name]}
                    for name, path in paths.items()},
                "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
