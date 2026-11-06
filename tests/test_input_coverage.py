import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from serein.data.availability import Availability
from serein.eval.input_coverage import (
    main,
    model_input_coverage,
    summarize_coverage,
    write_coverage,
)
from serein.features.table import MET_VARS, _model_features

DAY = date(2023, 8, 18)


@pytest.fixture
def points():
    return pd.DataFrame({"point_id": ["west", "east", "absent"], "gmt_offset": [-8., -5., -8.]})


def cache_for(points, *, source="hrrr", hours=range(49)):
    fields = ["smoke", *MET_VARS] if source == "hrrr" else ["pm25"]
    rows = [{"point_id": point, "issue_date": DAY,
             "valid_utc": pd.Timestamp(DAY) + pd.Timedelta(hours=12 + hour),
             "fhr": hour, "step": hour, "var": variable, "value": float(hour)}
            for point in points.point_id[:2] for variable in fields for hour in hours]
    return pd.DataFrame(rows)


def audit(cache, points, *, source="hrrr", leads=None):
    return model_input_coverage(cache, points, DAY, [0, 1, 2, 3] if leads is None else leads,
                                 source=source, cycle=12)


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
def test_audit_retains_every_requested_point_field_and_lead(points, source):
    result = audit(cache_for(points, source=source), points, source=source)
    assert set(result.point_id) == set(points.point_id)
    assert set(result.source) == {source}
    assert result.issue_date.eq(DAY).all()
    daily = result[result.aggregation.eq("local_day")]
    assert set(daily.lead) == {0, 1, 2, 3}
    assert daily.groupby(["point_id", "variable"]).size().eq(4).all()
    assert result.loc[result.point_id.eq("absent"), "status"].eq("missing_rows").all()
    assert result.loc[result.point_id.eq("absent"), ["mean", "max"]].isna().all().all()


@pytest.mark.parametrize("source,variable", [("hrrr", "smoke"), ("aqm", "pm25")])
def test_counts_follow_local_standard_days_and_preserve_unreachable_leads(points, source, variable):
    result = audit(cache_for(points, source=source), points, source=source).set_index(
        ["point_id", "variable", "aggregation", "lead"]).sort_index()
    west = result.loc[("west", variable, "local_day")]
    east = result.loc[("east", variable, "local_day")]
    assert west.n_finite.tolist() == [20, 24, 5, 0]
    assert east.n_finite.tolist() == [17, 24, 8, 0]
    assert west.status.tolist() == ["available", "available", "insufficient_samples", "missing_rows"]
    assert west.loc[0, "mean"] == pytest.approx(9.5)
    assert west.loc[1, "max"] == 43
    assert pd.isna(west.loc[2, "mean"])
    assert west.loc[3, "valid_date"] == pd.Timestamp("2023-08-21")


def test_weather_and_smoke_use_different_sample_thresholds(points):
    result = audit(cache_for(points, hours=range(4)), points, leads=[0])
    west = result.loc[result.point_id.eq("west") & result.aggregation.eq("local_day")].set_index("variable")
    assert west.loc["smoke", "min_count"] == 12
    assert west.loc["smoke", "status"] == "insufficient_samples"
    assert west.loc[MET_VARS, "min_count"].eq(4).all()
    assert west.loc[MET_VARS, "status"].eq("available").all()


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
def test_duplicate_samples_do_not_inflate_counts_or_weight_dense_hours(points, source):
    cache = cache_for(points, source=source, hours=range(12))
    duplicate = cache.loc[cache.fhr.eq(0)].copy()
    duplicate["value"] = 10.
    result = audit(pd.concat([cache, duplicate, duplicate]), points, source=source, leads=[0])
    first = result.loc[result.point_id.eq("west") & result.aggregation.eq("local_day")].iloc[0]
    assert first.n_finite == 12
    assert first.n_timestamps == 12
    assert first["mean"] == pytest.approx((sum(range(12)) + 20 / 3) / 12)


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
def test_nonfinite_samples_are_counted_but_not_used(points, source):
    cache = cache_for(points, source=source, hours=range(12))
    cache.loc[cache.point_id.eq("east"), "value"] = np.nan
    cache.loc[cache.point_id.eq("west") & cache.fhr.eq(0), "value"] = np.inf
    result = audit(cache, points, source=source, leads=[0])
    daily = result[result.aggregation.eq("local_day")]
    west = daily.loc[daily.point_id.eq("west")].iloc[0]
    east = daily.loc[daily.point_id.eq("east")].iloc[0]
    assert west.n_timestamps == east.n_timestamps == 12
    assert west.n_finite == 11
    assert east.n_finite == 0
    assert east.status == "no_finite_values"
    assert daily.loc[daily.variable.isin(["smoke", "pm25"]), "mean"].isna().all()


def test_missing_field_does_not_disappear(points):
    cache = cache_for(points)
    result = audit(cache.loc[cache["var"].ne("smoke")], points)
    smoke = result.loc[result.variable.eq("smoke")]
    assert len(smoke) == len(points) * 4
    assert smoke.status.eq("missing_rows").all()


def test_empty_cache_is_reported_without_required_columns(points):
    result = audit(pd.DataFrame(), points)
    assert result.status.eq("missing_rows").all()
    assert result.n_timestamps.eq(0).all()
    assert result.n_finite.eq(0).all()
    assert result[["mean", "max"]].isna().all().all()
    assert result["mean"].dtype == result["max"].dtype == np.dtype("float64")


def test_wind_at_issue_time_uses_the_pipeline_forecast_hour(points):
    cache = cache_for(points)
    duplicate = cache.loc[cache.fhr.eq(6) & cache["var"].eq("u10")].copy()
    duplicate["value"] = 10.
    result = audit(pd.concat([cache, duplicate]), points)
    wind = result.loc[result.aggregation.eq("issue_wind") & result.point_id.eq("west")].set_index("variable")
    assert wind.loc["u10", "mean"] == 8.
    assert wind.loc["v10", "mean"] == 6.
    assert wind.n_timestamps.eq(1).all()
    assert wind.n_finite.eq(1).all()
    assert wind.status.eq("available").all()


def test_equivalent_wind_timestamps_count_once(points):
    cache = cache_for(points, hours=[6])
    duplicate = cache.copy()
    cache["valid_utc"] = "2023-08-18T18:00Z"
    duplicate["valid_utc"] = "2023-08-18T11:00-07:00"
    report = audit(pd.concat([cache, duplicate]), points)
    wind = report.loc[report.aggregation.eq("issue_wind") & report.point_id.ne("absent")]
    assert wind.n_timestamps.eq(1).all()
    assert wind.n_finite.eq(1).all()


def test_unused_fields_with_duplicate_indexes_do_not_change_wind_counts(points):
    cache = cache_for(points, hours=[6])
    unused = cache.copy()
    unused["var"] = "unused"
    report = audit(pd.concat([cache, unused]), points)
    wind = report.loc[report.aggregation.eq("issue_wind") & report.point_id.ne("absent")]
    assert wind.status.eq("available").all()
    assert wind.n_timestamps.eq(1).all()


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
def test_audit_does_not_mutate_inputs(points, source):
    cache = cache_for(points, source=source)
    cache.attrs["sentinel"] = "original"
    original_cache, original_points = cache.copy(deep=True), points.copy(deep=True)
    audit(cache, points, source=source)
    pd.testing.assert_frame_equal(cache, original_cache)
    pd.testing.assert_frame_equal(points, original_points)
    assert cache.attrs == original_cache.attrs


def test_summary_preserves_every_point_and_missing_group(points):
    report = audit(cache_for(points), points)
    summary = summarize_coverage(report)
    assert sum(row["n_points"] for row in summary) == len(report)
    assert all(row["issue_date"] == DAY.isoformat() for row in summary)
    assert any(row["status"] == "missing_rows" for row in summary)


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
@pytest.mark.parametrize("column", ["point_id", "issue_date", "valid_utc", "value"])
def test_nonempty_cache_requires_hourly_schema(points, source, column):
    with pytest.raises(ValueError, match="missing columns"):
        audit(cache_for(points, source=source).drop(columns=column), points, source=source)


@pytest.mark.parametrize("source,step", [("hrrr", "fhr"), ("aqm", "step")])
@pytest.mark.parametrize("bad", [-1, .5, np.nan, np.inf, True])
def test_forecast_step_corruption_is_rejected(points, source, step, bad):
    cache = cache_for(points, source=source)
    cache[step] = bad
    with pytest.raises(ValueError, match="steps"):
        audit(cache, points, source=source)


@pytest.mark.parametrize("bad", [None, "2023-08-18 13:00", "2023-08-19 12:00"])
def test_corrupt_valid_times_are_not_scored_as_requested_inputs(points, bad):
    cache = cache_for(points, hours=[0])
    cache["valid_utc"] = bad
    with pytest.raises(ValueError, match="timestamps"):
        audit(cache, points)


@pytest.mark.parametrize("bad", [None, date(2023, 8, 17), "2023-08-18 01:00", "2023-08-18T00:00Z"])
def test_mismatched_cache_issue_dates_fail(points, bad):
    cache = cache_for(points)
    cache["issue_date"] = bad
    with pytest.raises(ValueError, match="issue dates"):
        audit(cache, points)


@pytest.mark.parametrize("source", ["hrrr", "aqm"])
def test_utc_equivalent_cache_times_are_accepted(points, source):
    cache = cache_for(points, source=source)
    cache["valid_utc"] = pd.to_datetime(cache.valid_utc).dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
    assert audit(cache, points, source=source).status.eq("available").any()


@pytest.mark.parametrize("leads", [[], [1, 1], [True], [-1], [.5], [np.nan]])
def test_invalid_requested_leads_fail(points, leads):
    with pytest.raises(ValueError, match="leads"):
        audit(pd.DataFrame(), points, leads=leads)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_invalid_point_offsets_fail(points, bad):
    points["gmt_offset"] = bad
    with pytest.raises(ValueError, match="offsets"):
        audit(pd.DataFrame(), points)


@pytest.mark.parametrize("bad", [None, "", " ", "west"])
def test_invalid_point_identifiers_fail(points, bad):
    points.loc[1, "point_id"] = bad
    with pytest.raises(ValueError, match="point IDs"):
        audit(pd.DataFrame(), points)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_audit_daily_values_match_real_feature_pipeline(points, monkeypatch, dtype):
    from serein.data import aqm, hrrr

    hr, aq = cache_for(points), cache_for(points, source="aqm")
    for frame in (hr, aq):
        frame["value"] = (frame.value * .1234567 + .0321).astype(dtype)
    hr.loc[hr.fhr.between(20, 26) & hr["var"].eq("smoke"), "value"] = np.nan
    monkeypatch.setattr(hrrr, "load_cycles", lambda *args, **kwargs: hr.copy())
    monkeypatch.setattr(aqm, "load_cycles", lambda *args, **kwargs: aq.copy())
    features = _model_features([DAY], points, points.set_index("point_id").gmt_offset,
                               "synthetic", Availability()).set_index("point_id")
    for source, cache in (("hrrr", hr), ("aqm", aq)):
        report = audit(cache, points, source=source)
        for row in report.itertuples():
            if row.point_id not in features.index:
                assert row.status == "missing_rows"
                continue
            if row.aggregation == "issue_wind":
                name = f"wind_now_{row.variable}"
                wanted = features.loc[row.point_id].get(name, np.nan)
                np.testing.assert_allclose(row.mean, wanted, rtol=0, atol=0, equal_nan=True)
            else:
                for statistic in ("mean", "max") if row.variable in {"smoke", "pm25"} else ("mean",):
                    stem = "aqm" if source == "aqm" else f"hrrr_{row.variable}"
                    name = f"{stem}_{statistic}_L{row.lead}" if row.variable in {"smoke", "pm25"} else f"{stem}_L{row.lead}"
                    wanted = features.loc[row.point_id].get(name, np.nan)
                    np.testing.assert_allclose(getattr(row, statistic), wanted,
                                               rtol=0, atol=0, equal_nan=True,
                                               err_msg=f"{source} {row.variable} {row.lead} {statistic}")


def test_fractional_standard_time_offsets_match_pipeline(points):
    points["gmt_offset"] = -7.5
    result = audit(cache_for(points, hours=range(24)), points, leads=[0, 1])
    smoke = result[result.point_id.eq("west") & result.variable.eq("smoke")].set_index("lead")
    assert smoke.n_timestamps.tolist() == [20, 4]


def test_a_history_day_keeps_its_own_calendar_label(points):
    day = DAY - timedelta(days=1)
    cache = cache_for(points)
    cache["issue_date"] = day
    cache["valid_utc"] -= pd.Timedelta(days=1)
    result = model_input_coverage(cache, points, day, [1], source="hrrr", cycle=12)
    assert result.issue_date.eq(day).all()
    assert result.loc[result.aggregation.eq("local_day"), "valid_date"].eq(pd.Timestamp(DAY)).all()


@pytest.fixture
def saved_inputs(points, tmp_path):
    points = points.assign(lat=38., lon=-121.)
    paths = [tmp_path / name for name in ("cache.parquet", "points.parquet", "audit")]
    cache = cache_for(points).drop(columns="issue_date")
    point_hash = hashlib.sha256(pd.util.hash_pandas_object(
        points[["point_id", "lat", "lon"]], index=False).to_numpy().tobytes()).hexdigest()
    cache.attrs.update(hrrr_request={"points": point_hash}, hrrr_complete=False)
    cache.to_parquet(paths[0], index=False)
    points.to_parquet(paths[1], index=False)
    return paths


def test_file_audit_records_exact_source_hashes_and_extraction_flags(saved_inputs):
    cache, points, output = saved_inputs
    write_coverage(cache, points, output, DAY, [0, 1, 2, 3], source="hrrr", cycle=12)
    report = pd.read_parquet(output / "input_coverage.parquet")
    metadata = json.loads((output / "summary.json").read_text())
    assert metadata["extraction_registry_match"] is True
    assert metadata["extraction_complete"] is False
    assert metadata["n_points"] == 3
    for name, path in (("cache", cache), ("points", points)):
        assert metadata["files"][name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert sum(row["n_points"] for row in metadata["coverage"]) == len(report)


@pytest.mark.parametrize("label", ["missing", "stale"])
def test_audit_exposes_missing_or_stale_registry_provenance(saved_inputs, label):
    cache, points, output = saved_inputs
    frame = pd.read_parquet(cache)
    frame.attrs = {} if label == "missing" else {"hrrr_request": {"points": "different"}}
    frame.to_parquet(cache, index=False)
    write_coverage(cache, points, output, DAY, [1], source="hrrr", cycle=12)
    metadata = json.loads((output / "summary.json").read_text())
    assert metadata["extraction_registry_match"] is (None if label == "missing" else False)


def test_cli_writes_readable_report_without_a_model(saved_inputs):
    cache, points, output = saved_inputs
    assert main(["--cache", str(cache), "--points", str(points), "--source", "hrrr",
                  "--date", DAY.isoformat(), "--cycle", "12", "--output", str(output)]) == 0
    assert not pd.read_parquet(output / "input_coverage.parquet").empty


def test_cli_refuses_existing_directory_before_reading_inputs(saved_inputs):
    cache, points, output = saved_inputs
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(SystemExit) as caught:
        main(["--cache", str(cache), "--points", str(points), "--source", "hrrr",
               "--date", DAY.isoformat(), "--cycle", "12", "--output", str(output)])
    assert caught.value.code == 2
    assert sentinel.read_text() == "original"
    assert list(output.iterdir()) == [sentinel]


def test_failed_validation_does_not_create_a_partial_audit(saved_inputs):
    cache, points, output = saved_inputs
    with pytest.raises(ValueError, match="leads"):
        write_coverage(cache, points, output, DAY, [-1], source="hrrr", cycle=12)
    assert not output.exists()


def test_missing_output_parent_is_reported(saved_inputs):
    cache, points, output = saved_inputs
    with pytest.raises(ValueError, match="parent"):
        write_coverage(cache, points, output / "missing", DAY, [1], source="hrrr", cycle=12)
    assert not output.exists()
