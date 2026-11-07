"""Inventory development NOAA extractions and their daily feature availability."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import io
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from serein.eval.episode_inputs import _choices
from serein.eval.input_coverage import model_input_coverage, summarize_coverage

FILE_COLUMNS = ["source", "issue_date", "cycle", "path", "sha256", "size_bytes", "file_status",
                "registry_status", "completion_status", "n_cache_rows", "n_foreign_points",
                "error_type", "error"]


def development_days(config: dict, years=None) -> list[date]:
    """Enumerate only configured training/calibration seasons, never test seasons."""
    split = config["split"]
    years_by_split = {name: set(_choices(split[name], f"{name} years", integers=True))
                      for name in ("train", "calibrate", "test")}
    if any(year < 1 or year > 9999 for values in years_by_split.values() for year in values):
        raise ValueError("split years must be calendar years in [1, 9999]")
    if any(years_by_split[a] & years_by_split[b]
           for a, b in (("train", "calibrate"), ("train", "test"), ("calibrate", "test"))):
        raise ValueError("training, calibration and reserved test years must not overlap")
    allowed = years_by_split["train"] | years_by_split["calibrate"]
    years = _choices(sorted(allowed) if years is None else years, "years", integers=True)
    if not set(years).issubset(allowed):
        raise ValueError("archive audits accept configured development years only")
    months = _choices(config["season_months"], "season months", integers=True)
    if any(month < 1 or month > 12 for month in months):
        raise ValueError("season months must be in [1, 12]")
    return sorted(date(year, month, day) for year in years for month in months
                  for day in range(1, calendar.monthrange(year, month)[1] + 1))


def _registry_status(cache: pd.DataFrame, source: str, digest: str) -> tuple[str, str]:
    request = cache.attrs.get("hrrr_request")
    stored = ((request.get("points") if isinstance(request, dict) else None)
              if source == "hrrr" else cache.attrs.get("aqm_points"))
    registry = "unknown" if stored is None else "match" if stored == digest else "mismatch"
    complete = cache.attrs.get("hrrr_complete" if source == "hrrr" else "aqm_samples_complete")
    completion = ("unknown" if complete is None else "declared_complete" if complete is True
                  else "declared_incomplete" if complete is False else "invalid")
    return registry, completion


def inspect_cache(path: Path, points: pd.DataFrame, day: date, leads: list[int], *,
                  source: str, cycle: int, point_digest: str,
                  issue_hour_utc: float = 18.5) -> tuple[dict, list[dict]]:
    """Record a bad file and continue; unavailable files never get zero-filled values.

    A corrupt file has unknown feature availability. A missing file has known
    missing rows. These are separate outcomes, even if neither can support a fit.
    """
    record = {"source": source, "issue_date": day.isoformat(), "cycle": cycle,
              "path": str(path.resolve()), "sha256": None, "size_bytes": None,
              "file_status": "missing", "registry_status": "unknown",
              "completion_status": "unknown", "n_cache_rows": None,
              "n_foreign_points": None, "error_type": None, "error": None}
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        report = model_input_coverage(pd.DataFrame(), points, day, leads,
                                       source=source, cycle=cycle, issue_hour_utc=issue_hour_utc)
        return record, summarize_coverage(report)
    except OSError as exc:
        record.update(file_status="unreadable", error_type=type(exc).__name__, error=str(exc))
        return record, []
    record.update(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    try:
        cache = pd.read_parquet(io.BytesIO(payload))
        record["n_cache_rows"] = len(cache)
        registry, completion = _registry_status(cache, source, point_digest)
        record.update(registry_status=registry, completion_status=completion)
        if "point_id" in cache:
            record["n_foreign_points"] = int(cache.loc[
                ~cache.point_id.isin(points.point_id), "point_id"].nunique(dropna=False))
        if "issue_date" not in cache:
            cache = cache.assign(issue_date=day)
        report = model_input_coverage(cache, points, day, leads, source=source, cycle=cycle,
                                       issue_hour_utc=issue_hour_utc)
    except (OSError, ValueError, TypeError, KeyError, pa.ArrowException) as exc:
        record.update(file_status="invalid", error_type=type(exc).__name__, error=str(exc))
        return record, []
    record["file_status"] = "inspected"
    return record, summarize_coverage(report)


def audit_archive(cache_root: Path, points_path: Path, output: Path, *, days,
                  leads, cycles: dict[str, int], issue_hour_utc: float = 18.5,
                  progress=None, metadata: dict | None = None) -> dict:
    """Read one cache at a time and save small summaries, retaining source hashes."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError("output parent must exist")
    if not cache_root.is_dir():
        raise ValueError("cache root must be an existing study interim directory")
    days = list(days)
    if (not days or len(set(days)) != len(days)
            or any(not isinstance(day, date) or isinstance(day, datetime) for day in days)):
        raise ValueError("days must be unique calendar dates")
    leads = _choices(leads, "leads", integers=True)
    if not cycles or not set(cycles).issubset({"hrrr", "aqm"}):
        raise ValueError("cycles must select hrrr and/or aqm")
    payload = points_path.read_bytes()
    points = pd.read_parquet(io.BytesIO(payload))
    required = {"point_id", "lat", "lon", "gmt_offset"}
    if not required.issubset(points):
        raise ValueError("registry requires point_id, lat, lon and gmt_offset")
    coordinates = points[["lat", "lon"]].to_numpy(dtype=float)
    if (not np.isfinite(coordinates).all()
            or (np.abs(coordinates) > [90, 180]).any()):
        raise ValueError("registry coordinates must be finite and within geographic bounds")
    for source, cycle in cycles.items():
        model_input_coverage(pd.DataFrame(), points, days[0], leads, source=source,
                               cycle=cycle, issue_hour_utc=issue_hour_utc)
    point_digest = hashlib.sha256(pd.util.hash_pandas_object(
        points[["point_id", "lat", "lon"]], index=False).to_numpy().tobytes()).hexdigest()
    folders = {"hrrr": "hrrr", "aqm": "aqm_bc"}
    manifest = {"report_version": 1, "status": "running", "cache_root": str(cache_root.resolve()),
                "points": str(points_path.resolve()), "points_sha256": hashlib.sha256(payload).hexdigest(),
                "registry_points_sha256": point_digest, "n_points": len(points),
                "dates": [day.isoformat() for day in sorted(days)], "leads": [int(v) for v in leads],
                "cycles": {name: int(cycle) for name, cycle in cycles.items()},
                "issue_hour_utc": issue_hour_utc, "request": metadata or {},
                "n_expected_files": len(days) * len(cycles), "n_inspected_files": 0,
                "interpretation": "offline extraction and input availability only; no posting-time or skill validation"}
    output.mkdir(exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records, file_counts, registry_counts = [], {}, {}
    try:
        with (output / "coverage.csv").open("x", encoding="utf-8", newline="") as stream:
            wrote_header = False
            for day in sorted(days):
                for source, cycle in cycles.items():
                    path = cache_root / folders[source] / f"{day:%Y%m%d}_{cycle:02d}z.parquet"
                    record, coverage = inspect_cache(
                        path, points, day, leads, source=source, cycle=cycle,
                        point_digest=point_digest, issue_hour_utc=issue_hour_utc,
                    )
                    records.append(record)
                    file_counts[record["file_status"]] = file_counts.get(record["file_status"], 0) + 1
                    registry_counts[record["registry_status"]] = registry_counts.get(record["registry_status"], 0) + 1
                    if coverage:
                        pd.DataFrame(coverage).to_csv(stream, index=False, header=not wrote_header)
                        wrote_header = True
                    if progress is not None:
                        progress(record)
            # An all-invalid archive still has a readable, empty coverage table.
            if not wrote_header:
                stream.write("source,issue_date,variable,lead,aggregation,status,n_points\n")
        pd.DataFrame(records, columns=FILE_COLUMNS).to_csv(output / "files.csv", index=False)
        manifest.update(status="complete", n_inspected_files=len(records),
                        file_status_counts=file_counts, registry_status_counts=registry_counts)
    except BaseException:
        # A stopped scan remains distinguishable from a complete report.
        pd.DataFrame(records, columns=FILE_COLUMNS).to_csv(output / "files.csv", index=False)
        manifest.update(status="interrupted", n_inspected_files=len(records),
                        file_status_counts=file_counts, registry_status_counts=registry_counts)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/west_us.yaml"))
    parser.add_argument("--cache-root", type=Path, required=True, help="study directory containing hrrr/ and aqm_bc/")
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    parser.add_argument("--year", type=int, action="append", dest="years")
    parser.add_argument("--source", action="append", dest="sources", choices=("hrrr", "aqm"))
    args = parser.parse_args(argv)
    try:
        config_bytes = args.config.read_bytes()
        config = yaml.safe_load(io.BytesIO(config_bytes))
        days = development_days(config, args.years)
        sources = _choices(args.sources or ["hrrr", "aqm"], "sources")
        cycles = {source: config["availability"][f"{source}_cycle"] for source in sources}
        def progress(record):
            print(f"{record['issue_date']} {record['source']}: {record['file_status']}", flush=True)
        manifest = audit_archive(
            args.cache_root, args.points, args.output, days=days, leads=config["leads"],
            cycles=cycles, issue_hour_utc=config["issue_hour_utc"], progress=progress,
            metadata={"config": str(args.config.resolve()),
                      "config_sha256": hashlib.sha256(config_bytes).hexdigest()},
        )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, pa.ArrowException) as exc:
        parser.error(str(exc))
    print(f"wrote {args.output}: {manifest['file_status_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
