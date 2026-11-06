"""Audit cached NOAA inputs using the feature pipeline's aggregation rules."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from serein.data.availability import validate_cycle_hour
from serein.features.daily import model_daily
from serein.features.table import MET_VARS


def _counts(frame: pd.DataFrame, offsets: pd.Series, *, minimum: int,
            extra: tuple = ()) -> pd.DataFrame:
    daily = model_daily(frame, offsets, min_count=minimum, group_extra=extra)
    present = model_daily(frame.assign(value=1.), offsets, min_count=1, group_extra=extra)
    keys = ["issue_date", "point_id", *extra, "lead"]
    return daily.rename(columns={"size": "n_finite"}).merge(
        present[keys + ["size"]].rename(columns={"size": "n_timestamps"}),
        on=keys, validate="one_to_one")


def model_input_coverage(cache: pd.DataFrame, points: pd.DataFrame, day: date,
                         leads: list[int], *, source: str, cycle: int,
                         issue_hour_utc: float = 18.5) -> pd.DataFrame:
    """Retain missing points, fields and local days rather than dropping them.

    Counts are distinct UTC timestamps, as in model_daily. Daily fields use the
    same 12-sample smoke/AQM and 4-sample weather thresholds as _model_features.
    Wind at issue time is counted separately from daily wind aggregates.
    """
    if source not in {"hrrr", "aqm"}:
        raise ValueError("source must be hrrr or aqm")
    validate_cycle_hour(cycle)
    if (isinstance(issue_hour_utc, (bool, np.bool_)) or not np.isfinite(issue_hour_utc)
            or not 0 <= issue_hour_utc < 24):
        raise ValueError("issue_hour_utc must be finite and in [0, 24)")
    if not isinstance(day, date) or isinstance(day, datetime):
        raise TypeError("day must be a calendar date")
    if (not leads or len(set(leads)) != len(leads)
            or any(isinstance(lead, (bool, np.bool_))
                   or not isinstance(lead, (int, np.integer)) or lead < 0 for lead in leads)):
        raise ValueError("leads must be distinct nonnegative integers")
    required = {"point_id", "gmt_offset"}
    if not required.issubset(points):
        raise ValueError("points require point_id and gmt_offset")
    if (points.empty or points.point_id.duplicated().any()
            or not points.point_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError("point IDs must be distinct nonempty strings")
    offsets = points.set_index("point_id").gmt_offset.astype(float)
    if not np.isfinite(offsets).all():
        raise ValueError("point offsets must be finite")
    fields = ["smoke", *MET_VARS] if source == "hrrr" else ["pm25"]
    expected = pd.MultiIndex.from_product(
        [points.point_id, fields, leads], names=["point_id", "variable", "lead"]).to_frame(index=False)
    expected["aggregation"] = "local_day"
    expected["min_count"] = np.where(expected.variable.isin(["smoke", "pm25"]), 12, 4)
    if source == "hrrr":
        wind = pd.MultiIndex.from_product(
            [points.point_id, ["u10", "v10"]], names=["point_id", "variable"]).to_frame(index=False)
        wind["lead"], wind["aggregation"], wind["min_count"] = 0, "issue_wind", 1
        expected = pd.concat([expected, wind], ignore_index=True)
    keys = ["point_id", "variable", "lead", "aggregation"]
    measured = pd.DataFrame(columns=keys + ["n_finite", "n_timestamps", "mean", "max"])
    if not cache.empty:
        step = "fhr" if source == "hrrr" else "step"
        required = {"point_id", "issue_date", "valid_utc", "value", step}
        if source == "hrrr":
            required.add("var")
        if not required.issubset(cache):
            raise ValueError(f"{source} cache missing columns: {', '.join(sorted(required - set(cache)))}")
        frame = cache.loc[cache.point_id.isin(points.point_id)].copy()
        if not frame.empty:
            issue = pd.to_datetime(frame.issue_date, format="mixed", errors="raise")
            if (isinstance(issue.dtype, pd.DatetimeTZDtype) or issue.isna().any()
                    or not issue.eq(pd.Timestamp(day)).all()):
                raise ValueError("cache issue dates must match the requested calendar date")
            hours = pd.to_numeric(frame[step], errors="raise")
            if (frame[step].map(lambda value: isinstance(value, (bool, np.bool_))).any()
                    or not np.isfinite(hours).all() or (hours < 0).any() or (hours % 1 != 0).any()):
                raise ValueError("cache forecast steps must be nonnegative integers")
            valid = pd.to_datetime(frame.valid_utc, format="mixed", utc=True, errors="raise")
            wanted = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=cycle) + pd.to_timedelta(hours, unit="h")
            if valid.isna().any() or not valid.eq(wanted).all():
                raise ValueError("cache timestamps must match their issue date, cycle and forecast step")
            frame["valid_utc"], frame[step] = valid, hours
            frame["value"] = pd.to_numeric(frame.value, errors="raise")
            frame["var"] = frame["var"].astype(str) if source == "hrrr" else "pm25"
            frame = frame.loc[frame["var"].isin(fields)]
            if source == "hrrr":
                daily = pd.concat([
                    _counts(frame.loc[frame["var"].eq("smoke")], offsets, minimum=12).assign(var="smoke"),
                    _counts(frame.loc[frame["var"].isin(MET_VARS)], offsets,
                             minimum=4, extra=("var",)),
                ], ignore_index=True)
            else:
                daily = _counts(frame, offsets, minimum=12).assign(var="pm25")
            daily["aggregation"] = "local_day"
            parts = [daily]
            if source == "hrrr":
                now = frame.loc[frame[step].eq(int(issue_hour_utc - cycle))
                                & frame["var"].isin(["u10", "v10"])].copy()
                now["value"] = now.value.where(np.isfinite(now.value))
                grouped = now.groupby(["point_id", "var"], observed=True)
                wind = grouped.value.agg(mean="mean", max="max", n_finite="count").reset_index()
                wind["n_timestamps"] = grouped.valid_utc.nunique().to_numpy()
                wind["n_finite"] = wind.n_finite.clip(upper=1)
                wind["max"] = wind["mean"]
                wind["lead"], wind["aggregation"] = 0, "issue_wind"
                parts.append(wind)
            measured = pd.concat(parts, ignore_index=True).rename(columns={"var": "variable"})
    report = expected.merge(measured.drop(columns="issue_date", errors="ignore"),
                             on=keys, how="left", validate="one_to_one")
    for column in ("n_timestamps", "n_finite"):
        report[column] = pd.to_numeric(report[column], errors="raise").fillna(0).astype("int64")
    for column in ("mean", "max"):
        report[column] = pd.to_numeric(report[column], errors="raise").astype("float64")
    report["status"] = np.select(
        [report.n_timestamps.eq(0), report.n_finite.eq(0), report.n_finite.lt(report.min_count)],
        ["missing_rows", "no_finite_values", "insufficient_samples"], default="available")
    report.loc[report.status.ne("available"), ["mean", "max"]] = np.nan
    report["source"], report["issue_date"] = source, day
    report["valid_date"] = pd.Timestamp(day) + pd.to_timedelta(report.lead, unit="D")
    return report[["source", "issue_date", *keys, "valid_date", "min_count",
                   "n_timestamps", "n_finite", "mean", "max", "status"]]


def summarize_coverage(report: pd.DataFrame) -> list[dict]:
    """Count points per source, field, input day and status without hiding gaps."""
    keys = ["source", "issue_date", "variable", "lead", "aggregation", "status"]
    summary = report.groupby(keys, sort=True, observed=True).size().rename("n_points").reset_index()
    summary["issue_date"] = summary.issue_date.map(lambda value: value.isoformat())
    return summary.to_dict(orient="records")


def write_coverage(cache_path: Path, points_path: Path, output: Path, day: date,
                   leads: list[int], *, source: str, cycle: int,
                   issue_hour_utc: float = 18.5) -> None:
    """Audit explicit local files without loading a model or downloading data."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent does not exist: {output.parent}")
    payloads = {"cache": cache_path.read_bytes(), "points": points_path.read_bytes()}
    cache, points = [pd.read_parquet(io.BytesIO(payloads[name])) for name in ("cache", "points")]
    # Stored extractions label issue dates when loaded, not when saved.
    if "issue_date" not in cache:
        cache = cache.assign(issue_date=day)
    report = model_input_coverage(cache, points, day, leads, source=source, cycle=cycle,
                                   issue_hour_utc=issue_hour_utc)
    point_hash = hashlib.sha256(pd.util.hash_pandas_object(
        points[["point_id", "lat", "lon"]], index=False).to_numpy().tobytes()).hexdigest()
    request = cache.attrs.get("hrrr_request")
    extracted_hash = ((request.get("points") if isinstance(request, dict) else None) if source == "hrrr"
                      else cache.attrs.get("aqm_points"))
    metadata = {"source": source, "issue_date": day.isoformat(), "cycle": cycle,
                "leads": [int(lead) for lead in leads], "issue_hour_utc": issue_hour_utc,
                "n_points": len(points), "registry_points_sha256": point_hash,
                "extraction_registry_match": None if extracted_hash is None else extracted_hash == point_hash,
                "extraction_complete": cache.attrs.get(
                    "hrrr_complete" if source == "hrrr" else "aqm_samples_complete"),
                "files": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(payloads[name]).hexdigest()}
                          for name, path in (("cache", cache_path), ("points", points_path))},
                "coverage": summarize_coverage(report),
                "interpretation": "NOAA feature availability only; no forecast skill or observational coverage assessed"}
    output.mkdir(exist_ok=False)
    report.to_parquet(output / "input_coverage.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="one local NOAA extraction Parquet")
    parser.add_argument("--points", type=Path, required=True, help="point registry used for extraction")
    parser.add_argument("--source", choices=("hrrr", "aqm"), required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="extraction issue date")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--leads", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--issue-hour-utc", type=float, default=18.5)
    parser.add_argument("--output", type=Path, required=True, help="new directory under an existing parent")
    args = parser.parse_args(argv)
    try:
        write_coverage(args.cache, args.points, args.output, args.date, args.leads,
                         source=args.source, cycle=args.cycle, issue_hour_utc=args.issue_hour_utc)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Wrote NOAA input coverage to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
