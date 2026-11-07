import hashlib
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
import yaml

from serein.eval.cache_archive import audit_archive, development_days, inspect_cache, main
from serein.eval.input_coverage import model_input_coverage, summarize_coverage

DAY = date(2023, 8, 18)


def study():
    return {"split": {"train": [2021, 2022], "calibrate": [2023], "test": [2024, 2025]},
            "season_months": [8], "leads": [0, 1], "issue_hour_utc": 18.5,
            "availability": {"hrrr_cycle": 12, "aqm_cycle": 12}}


@pytest.fixture
def archive(tmp_path):
    points = pd.DataFrame({"point_id": ["A", "B"], "lat": [38., 39.],
                           "lon": [-121., -122.], "gmt_offset": [-8., -8.]})
    points_path = tmp_path / "points.parquet"
    points.to_parquet(points_path, index=False)
    root = tmp_path / "interim"
    (root / "hrrr").mkdir(parents=True)
    (root / "aqm_bc").mkdir()
    digest = hashlib.sha256(pd.util.hash_pandas_object(
        points[["point_id", "lat", "lon"]], index=False).to_numpy().tobytes()).hexdigest()
    return points, points_path, root, digest, tmp_path / "report"


def hourly(points, source, day=DAY):
    rows = [{"point_id": point, "valid_utc": pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=12 + h),
             "fhr" if source == "hrrr" else "step": h, "value": float(h)}
            for point in points.point_id for h in range(49)]
    table = pd.DataFrame(rows)
    if source == "hrrr":
        table["var"] = "smoke"
    return table


def test_development_dates_cover_only_requested_seasons_and_years():
    config = study()
    days = development_days(config)
    expected = [d.date() for year in (2021, 2022, 2023)
                for d in pd.date_range(f"{year}-08-01", f"{year}-08-31")]
    assert days == expected
    assert development_days(config, [2023]) == [day for day in days if day.year == 2023]
    assert not set(config["split"]["test"]) & {day.year for day in days}


def test_development_calendar_handles_leap_year_and_sorts_seasons():
    config = study()
    config["split"]["train"] = [2020]
    config["season_months"] = [3, 2]
    days = development_days(config, [2020])
    assert date(2020, 2, 29) in days
    assert days == sorted(days)
    assert len(days) == len(pd.date_range("2020-02-01", "2020-03-31"))


@pytest.mark.parametrize("years", [[2024], [2025], [2026], [2023, 2023], [], [True], [2023.]])
def test_reserved_or_invalid_year_selection_is_rejected(years):
    with pytest.raises(ValueError):
        development_days(study(), years)


@pytest.mark.parametrize("change", ["train-test overlap", "calibrate-test overlap", "train-calibrate overlap",
                                    "string test year", "bool year", "bad month", "duplicate month", "year zero"])
def test_invalid_study_calendar_is_rejected(change):
    config = study()
    if change == "train-test overlap":
        config["split"]["train"].append(2024)
    elif change == "calibrate-test overlap":
        config["split"]["calibrate"].append(2024)
    elif change == "train-calibrate overlap":
        config["split"]["train"].append(2023)
    elif change == "string test year":
        config["split"]["test"] = ["2024"]
    elif change == "bool year":
        config["split"]["train"] = [True]
    elif change == "bad month":
        config["season_months"] = [0]
    elif change == "duplicate month":
        config["season_months"] = [8, 8]
    else:
        config["split"]["train"] = [0]
    with pytest.raises(ValueError):
        development_days(config)


@pytest.mark.parametrize("source,folder", [("hrrr", "hrrr"), ("aqm", "aqm_bc")])
def test_inspection_matches_single_file_coverage_and_preserves_bytes(archive, source, folder):
    points, _, root, digest, _ = archive
    path = root / folder / "20230818_12z.parquet"
    table = hourly(points, source)
    if source == "hrrr":
        table.attrs.update(hrrr_request={"points": digest}, hrrr_complete=True)
    else:
        table.attrs.update(aqm_points=digest, aqm_samples_complete=True)
    table.to_parquet(path, index=False)
    before = path.read_bytes()
    record, coverage = inspect_cache(path, points, DAY, [0, 1], source=source,
                                     cycle=12, point_digest=digest)
    expected = summarize_coverage(model_input_coverage(table.assign(issue_date=DAY), points,
                                                       DAY, [0, 1], source=source, cycle=12))
    assert coverage == expected
    assert record["file_status"] == "inspected"
    assert record["registry_status"] == "match"
    assert record["completion_status"] == "declared_complete"
    assert record["n_cache_rows"] == len(table)
    assert record["n_foreign_points"] == 0
    assert record["sha256"] == hashlib.sha256(before).hexdigest()
    assert record["size_bytes"] == len(before)
    assert path.read_bytes() == before


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
def test_missing_cache_has_explicit_missing_coverage(archive, source):
    points, _, root, digest, _ = archive
    record, coverage = inspect_cache(root / "absent.parquet", points, DAY, [0, 1],
                                     source=source, cycle=12, point_digest=digest)
    assert record["file_status"] == "missing"
    assert record["sha256"] is None and record["n_cache_rows"] is None
    assert {row["status"] for row in coverage} == {"missing_rows"}
    expected = model_input_coverage(pd.DataFrame(), points, DAY, [0, 1], source=source, cycle=12)
    assert sum(row["n_points"] for row in coverage) == len(expected)


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
@pytest.mark.parametrize("kind", ["legacy", "mismatch", "incomplete", "invalid completion"])
def test_provenance_is_separate_from_feature_coverage(archive, source, kind):
    points, _, root, digest, _ = archive
    table = hourly(points, source)
    if kind != "legacy":
        stored = "another-registry" if kind == "mismatch" else digest
        complete = False if kind == "incomplete" else "yes" if kind == "invalid completion" else True
        if source == "hrrr":
            table.attrs.update(hrrr_request={"points": stored}, hrrr_complete=complete)
        else:
            table.attrs.update(aqm_points=stored, aqm_samples_complete=complete)
    path = root / "test.parquet"
    table.to_parquet(path, index=False)
    record, coverage = inspect_cache(path, points, DAY, [0], source=source, cycle=12, point_digest=digest)
    expected_registry = "unknown" if kind == "legacy" else "mismatch" if kind == "mismatch" else "match"
    assert record["registry_status"] == expected_registry
    assert record["completion_status"] == {"legacy": "unknown", "mismatch": "declared_complete",
                                            "incomplete": "declared_incomplete",
                                            "invalid completion": "invalid"}[kind]
    assert record["file_status"] == "inspected"
    assert any(row["status"] == "available" for row in coverage)


@pytest.mark.parametrize("kind", ["corrupt bytes", "missing schema", "wrong timestamp", "wrong issue date"])
def test_invalid_cache_has_unknown_coverage_and_original_error(archive, kind):
    points, _, root, digest, _ = archive
    path = root / "test.parquet"
    if kind == "corrupt bytes":
        path.write_bytes(b"not a parquet file")
    else:
        table = hourly(points, "aqm")
        if kind == "missing schema":
            table = table.drop(columns="step")
        elif kind == "wrong timestamp":
            table["valid_utc"] += pd.Timedelta(hours=1)
        else:
            table["issue_date"] = date(2023, 8, 19)
        table.to_parquet(path, index=False)
    before = path.read_bytes()
    record, coverage = inspect_cache(path, points, DAY, [0], source="aqm", cycle=12, point_digest=digest)
    assert record["file_status"] == "invalid"
    assert record["error_type"] and record["error"]
    assert coverage == []
    assert record["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_foreign_point_ids_are_counted_without_becoming_registry_support(archive):
    points, _, root, digest, _ = archive
    table = hourly(points, "aqm")
    table["point_id"] = "FOREIGN"
    path = root / "test.parquet"
    table.to_parquet(path, index=False)
    record, coverage = inspect_cache(path, points, DAY, [0], source="aqm", cycle=12, point_digest=digest)
    assert record["n_foreign_points"] == 1
    assert {row["status"] for row in coverage} == {"missing_rows"}


def test_unreadable_cache_is_distinct_from_absent_cache(archive):
    points, _, root, digest, _ = archive
    record, coverage = inspect_cache(root, points, DAY, [0], source="aqm", cycle=12, point_digest=digest)
    assert record["file_status"] == "unreadable"
    assert coverage == []


def test_archive_retains_missing_invalid_and_legacy_files(archive):
    points, points_path, root, digest, output = archive
    hourly(points, "hrrr").to_parquet(root / "hrrr/20230818_12z.parquet", index=False)
    (root / "aqm_bc/20230818_12z.parquet").write_bytes(b"truncated")
    dates = [DAY, date(2023, 8, 19)]
    manifest = audit_archive(root, points_path, output, days=dates, leads=[0, 1],
                              cycles={"hrrr": 12, "aqm": 12})
    files = pd.read_csv(output / "files.csv")
    coverage = pd.read_csv(output / "coverage.csv")
    assert manifest["status"] == "complete"
    assert manifest["n_expected_files"] == manifest["n_inspected_files"] == len(dates) * 2
    assert manifest["file_status_counts"] == {"inspected": 1, "invalid": 1, "missing": 2}
    assert manifest["registry_status_counts"] == {"unknown": len(files)}
    assert manifest["registry_points_sha256"] == digest
    assert manifest["points_sha256"] == hashlib.sha256(points_path.read_bytes()).hexdigest()
    assert not ((coverage.source == "aqm") & (coverage.issue_date == DAY.isoformat())).any()
    missing = coverage.loc[coverage.issue_date.eq("2023-08-19")]
    assert missing.status.eq("missing_rows").all()
    assert len(list(root.rglob("*.parquet"))) == 2


def test_all_invalid_archive_still_writes_a_readable_coverage_schema(archive):
    _, points_path, root, _, output = archive
    (root / "aqm_bc/20230818_12z.parquet").write_bytes(b"truncated")
    result = audit_archive(root, points_path, output, days=[DAY], leads=[0], cycles={"aqm": 12})
    assert pd.read_csv(output / "coverage.csv").empty
    assert result["file_status_counts"] == {"invalid": 1}


def test_interruption_preserves_partial_inventory_with_incomplete_manifest(archive):
    _, points_path, root, _, output = archive
    def interrupt(record):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        audit_archive(root, points_path, output, days=[DAY, date(2023, 8, 19)],
                      leads=[0], cycles={"aqm": 12}, progress=interrupt)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["n_expected_files"] == 2
    assert manifest["n_inspected_files"] == 1
    assert len(pd.read_csv(output / "files.csv")) == 1
    assert not pd.read_csv(output / "coverage.csv").empty


def test_interruption_before_first_result_keeps_a_readable_empty_inventory(archive, monkeypatch):
    _, points_path, root, _, output = archive
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr("serein.eval.cache_archive.inspect_cache", interrupt)
    with pytest.raises(KeyboardInterrupt):
        audit_archive(root, points_path, output, days=[DAY], leads=[0], cycles={"aqm": 12})
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["n_inspected_files"] == 0
    files = pd.read_csv(output / "files.csv")
    assert files.empty
    assert "file_status" in files


@pytest.mark.parametrize("kwargs", [{"days": []}, {"days": [DAY, DAY]}, {"days": [datetime(2023, 8, 18)]},
                                    {"leads": []}, {"leads": [True]}, {"cycles": {}},
                                    {"cycles": {"unknown": 12}}, {"cycles": {"aqm": 24}},
                                    {"issue_hour_utc": np.nan}])
def test_bad_requests_fail_before_creating_output(archive, kwargs):
    _, points_path, root, _, output = archive
    options = {"days": [DAY], "leads": [0], "cycles": {"aqm": 12}, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        audit_archive(root, points_path, output, **options)
    assert not output.exists()


@pytest.mark.parametrize("change", ["missing lat", "bad lat", "duplicate IDs", "missing offset"])
def test_invalid_registry_fails_before_creating_output(archive, change):
    points, points_path, root, _, output = archive
    if change == "missing lat":
        points = points.drop(columns="lat")
    elif change == "bad lat":
        points.loc[0, "lat"] = 91.
    elif change == "duplicate IDs":
        points.loc[0, "point_id"] = "B"
    else:
        points = points.drop(columns="gmt_offset")
    points.to_parquet(points_path, index=False)
    with pytest.raises(ValueError):
        audit_archive(root, points_path, output, days=[DAY], leads=[0], cycles={"aqm": 12})
    assert not output.exists()


def test_cli_only_scans_selected_development_year_and_preserves_config_hash(archive, tmp_path):
    _, points_path, root, _, output = archive
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump(study()))
    args = ["--config", str(config), "--cache-root", str(root), "--points", str(points_path),
            "--output", str(output), "--source", "aqm", "--year", "2023"]
    assert main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["n_inspected_files"] == len(pd.date_range("2023-08-01", "2023-08-31"))
    assert manifest["request"]["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert {value[:4] for value in manifest["dates"]} == {"2023"}
    assert manifest["cycles"] == {"aqm": 12}


@pytest.mark.parametrize("kind", ["reserved year", "existing output", "bad yaml", "missing root"])
def test_cli_refuses_unsafe_or_invalid_requests_without_overwriting(archive, tmp_path, kind):
    _, points_path, root, _, output = archive
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump(study()))
    extras = []
    if kind == "reserved year":
        extras = ["--year", "2024"]
    elif kind == "existing output":
        output.mkdir()
        (output / "sentinel").write_text("keep")
    elif kind == "bad yaml":
        config.write_text("split: [unterminated")
    else:
        root = root / "absent"
    with pytest.raises(SystemExit) as error:
        main(["--config", str(config), "--cache-root", str(root), "--points", str(points_path),
              "--output", str(output), *extras])
    assert error.value.code == 2
    if kind == "existing output":
        assert (output / "sentinel").read_text() == "keep"
    else:
        assert not output.exists()
