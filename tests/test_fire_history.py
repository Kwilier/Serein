from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from serein.features.fire_history import augment_fire_history, fire_history_features, main

ISSUE = pd.Timestamp("2023-07-03 18:30", tz="UTC")


def inputs():
    fires = pd.DataFrame({
        "source": ["VIIRS"] * 5,
        "latitude": [38., 38., 38., 38., 45.],
        "longitude": [-121.] * 5,
        "time_utc": [ISSUE - pd.Timedelta(hours=h) for h in (48, 24, 12, 0, 1)],
        "available_utc": [ISSUE - pd.Timedelta(hours=h) for h in (47, 23, 11, 0, 0)],
        "frp": [1000., 5., 10., 20., 900.],
    })
    coverage = pd.DataFrame({"source": ["VIIRS"], "start_utc": [ISSUE - pd.Timedelta(days=2)],
                             "end_utc": [ISSUE], "available_utc": [ISSUE],
                             "lat_min": [-90.], "lat_max": [90.],
                             "lon_min": [-180.], "lon_max": [180.]})
    points = pd.DataFrame({"point_id": ["near", "far"], "lat": [38., 0.], "lon": [-121., 0.]},
                          index=[8, 2])
    return fires, coverage, points


def build(fires=None, coverage=None, points=None, **kwargs):
    defaults = inputs()
    return fire_history_features(
        defaults[0] if fires is None else fires,
        defaults[1] if coverage is None else coverage,
        defaults[2] if points is None else points, ISSUE, source="VIIRS", **kwargs)


def test_adjacent_windows_boundaries_distance_and_signed_growth():
    result = build()
    assert result.point_id.tolist() == ["near", "far"]
    np.testing.assert_array_equal(result.fire_recent_count, [2, 0])
    np.testing.assert_array_equal(result.fire_previous_count, [1, 0])
    np.testing.assert_array_equal(result.fire_recent_frp, [30, 0])
    np.testing.assert_array_equal(result.fire_previous_frp, [5, 0])
    np.testing.assert_allclose(result.fire_frp_log_change, [np.log(31 / 6), 0])
    assert result.fire_latest_age_hours.iloc[0] == 0
    assert np.isnan(result.fire_latest_age_hours.iloc[1])
    assert result.fire_recent_complete.all() and result.fire_previous_complete.all()


def test_empty_archive_is_zero_only_with_declared_coverage():
    fires, coverage, _ = inputs()
    complete = build(fires=fires.iloc[:0])
    missing = build(fires=fires.iloc[:0], coverage=coverage.iloc[:0])
    assert complete.fire_recent_frp.eq(0).all()
    assert complete.fire_recent_count.eq(0).all()
    assert complete.fire_latest_age_hours.isna().all()
    assert missing.fire_recent_frp.isna().all()
    assert missing.fire_recent_count.isna().all()
    assert missing.fire_frp_log_change.isna().all()
    assert not missing.fire_recent_complete.any()


def test_source_availability_and_future_acquisitions_do_not_leak():
    fires, coverage, _ = inputs()
    expected = build()
    later = fires.iloc[[2]].copy()
    later["available_utc"] = ISSUE + pd.Timedelta(seconds=1)
    later["frp"] = 99999
    future = later.copy()
    future["time_utc"] = ISSUE + pd.Timedelta(seconds=1)
    future["available_utc"] = ISSUE + pd.Timedelta(seconds=2)
    other = later.copy()
    other["source"] = "MODIS"
    other["available_utc"] = ISSUE
    result = build(fires=pd.concat([fires, later, future, other]))
    pd.testing.assert_frame_equal(result, expected)
    delayed = coverage.assign(available_utc=ISSUE + pd.Timedelta(seconds=1))
    assert build(coverage=delayed).fire_recent_frp.isna().all()


def test_coverage_union_gap_and_window_specific_missingness():
    _, coverage, _ = inputs()
    first = coverage.assign(end_utc=ISSUE - pd.Timedelta(days=1))
    second = coverage.assign(start_utc=ISSUE - pd.Timedelta(days=1))
    pd.testing.assert_frame_equal(build(coverage=pd.concat([second, first, first])), build())
    second["start_utc"] += pd.Timedelta(microseconds=1)
    gap = build(coverage=pd.concat([first, second]))
    assert gap.fire_previous_complete.all()
    assert not gap.fire_recent_complete.any()
    assert gap.fire_previous_frp.eq(build().fire_previous_frp).all()
    assert gap.fire_recent_frp.isna().all() and gap.fire_frp_log_change.isna().all()


def test_duplicates_count_once_and_conflicting_power_rejected():
    fires, _, _ = inputs()
    pd.testing.assert_frame_equal(build(fires=pd.concat([fires, fires])), build())
    conflict = fires.iloc[[2]].assign(frp=11)
    with pytest.raises(ValueError, match="conflicting FRP"):
        build(fires=pd.concat([fires, conflict]))


def test_missing_frp_keeps_counts_and_age_without_inventing_power():
    fires, _, _ = inputs()
    fires.loc[2, "frp"] = np.nan
    result = build(fires=fires)
    assert result.fire_recent_count.tolist() == [2, 0]
    assert result.fire_latest_age_hours.iloc[0] == 0
    assert np.isnan(result.fire_recent_frp.iloc[0])
    assert result.fire_recent_frp.iloc[1] == 0
    assert np.isnan(result.fire_frp_log_change.iloc[0])


def test_decline_has_negative_change_and_inputs_stay_unchanged():
    fires, coverage, points = inputs()
    fires.loc[1, "frp"] = 100
    original = [x.copy(deep=True) for x in (fires, coverage, points)]
    result = build(fires, coverage, points)
    assert result.fire_frp_log_change.iloc[0] < 0
    for before, after in zip(original, (fires, coverage, points)):
        pd.testing.assert_frame_equal(before, after)


def test_numeric_strings_are_normalized_without_mutating_inputs():
    fires, coverage, points = inputs()
    for column in ("latitude", "longitude", "frp"):
        fires[column] = fires[column].astype(str)
    for column in ("lat_min", "lat_max", "lon_min", "lon_max"):
        coverage[column] = coverage[column].astype(str)
    for column in ("lat", "lon"):
        points[column] = points[column].astype(str)
    before = [frame.copy(deep=True) for frame in (fires, coverage, points)]
    pd.testing.assert_frame_equal(build(fires, coverage, points), build())
    for original, current in zip(before, (fires, coverage, points)):
        pd.testing.assert_frame_equal(original, current)


def test_nullable_missing_frp_remains_missing():
    fires, _, _ = inputs()
    fires["frp"] = fires.frp.astype("Float64")
    fires.loc[2, "frp"] = pd.NA
    result = build(fires=fires)
    assert np.isnan(result.fire_recent_frp.iloc[0])
    assert result.fire_recent_count.iloc[0] == 2


@pytest.mark.parametrize("field", ["latitude", "longitude", "frp", "lat", "lon", "lat_min"])
def test_complex_input_is_rejected_without_discarding_imaginary_part(field):
    fires, coverage, points = inputs()
    frame = fires if field in fires else points if field in points else coverage
    frame[field] = frame[field].astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        build(fires, coverage, points)


def test_utc_offsets_and_naive_times_represent_same_instants():
    fires, coverage, points = inputs()
    for column in ("time_utc", "available_utc"):
        fires[column] = fires[column].dt.tz_convert("America/Los_Angeles")
    for column in ("start_utc", "end_utc", "available_utc"):
        coverage[column] = coverage[column].dt.tz_localize(None)
    pd.testing.assert_frame_equal(build(fires, coverage, points), build())
    alternate = fire_history_features(fires, coverage, points, "2023-07-03 11:30-07:00",
                                      source="VIIRS")
    pd.testing.assert_frame_equal(alternate, build())


@pytest.mark.parametrize("name", ["window_hours", "radius_km"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True])
def test_invalid_parameters(name, value):
    with pytest.raises(ValueError, match=name):
        build(**{name: value})


@pytest.mark.parametrize("column,value,match", [
    ("latitude", 91, "latitude"), ("longitude", -181, "longitude"),
    ("latitude", np.nan, "latitude"), ("frp", -1, "frp"), ("frp", np.inf, "frp"),
    ("time_utc", pd.NaT, "missing timestamps"),
    ("available_utc", pd.NaT, "missing timestamps"),
    ("available_utc", ISSUE - pd.Timedelta(days=9), "before acquisition"),
])
def test_invalid_active_detections(column, value, match):
    fires, _, _ = inputs()
    fires.loc[2, column] = value
    with pytest.raises(ValueError, match=match):
        build(fires=fires)


@pytest.mark.parametrize("column,value,match", [
    ("start_utc", ISSUE, "must follow"),
    ("end_utc", ISSUE + pd.Timedelta(hours=1), "before its interval ends"),
    ("available_utc", pd.NaT, "missing timestamps"),
])
def test_invalid_coverage(column, value, match):
    _, coverage, _ = inputs()
    coverage.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        build(coverage=coverage)


def test_empty_points_and_unknown_source():
    fires, coverage, points = inputs()
    assert build(points=points.iloc[:0]).empty
    result = fire_history_features(fires, coverage, points, ISSUE, source="unknown")
    assert not result.fire_recent_complete.any()
    assert result.fire_recent_frp.isna().all()


def test_acquisition_lag_moves_both_windows_but_age_uses_issue():
    fires, coverage, _ = inputs()
    coverage["start_utc"] -= pd.Timedelta(hours=6)
    result = build(fires, coverage, acquisition_lag_hours=6)
    assert result.fire_recent_count.iloc[0] == 2
    assert result.fire_recent_frp.iloc[0] == 15
    assert result.fire_previous_frp.iloc[0] == 1000
    assert result.fire_latest_age_hours.iloc[0] == 12
    assert result.window_end_utc.eq(ISSUE - pd.Timedelta(hours=6)).all()


def test_coverage_requires_full_radius_around_each_point():
    fires, coverage, _ = inputs()
    regional = coverage.assign(lat_min=34., lat_max=42., lon_min=-125., lon_max=-117.)
    result = build(fires, regional)
    assert result.fire_recent_complete.tolist() == [True, False]
    assert result.fire_recent_frp.iloc[0] == 30
    assert np.isnan(result.fire_recent_frp.iloc[1])
    narrow = regional.assign(lon_min=-122.)
    result = build(fires, narrow)
    assert not result.fire_recent_complete.any()
    assert result.fire_recent_count.isna().all()


def test_temporal_union_must_cover_the_same_spatial_footprint():
    _, coverage, _ = inputs()
    regional = coverage.assign(lat_min=34., lat_max=42., lon_min=-125., lon_max=-117.,
                               end_utc=ISSUE - pd.Timedelta(hours=12))
    elsewhere = coverage.assign(start_utc=ISSUE - pd.Timedelta(hours=12),
                                lat_min=-10., lat_max=10., lon_min=-10., lon_max=10.)
    result = build(coverage=pd.concat([regional, elsewhere]))
    assert not result.fire_recent_complete.any()
    assert result.fire_previous_complete.tolist() == [True, False]


def test_different_rectangles_can_join_time_coverage_for_the_same_point():
    _, coverage, _ = inputs()
    first = coverage.assign(lat_min=34., lat_max=42., lon_min=-125., lon_max=-117.,
                            end_utc=ISSUE - pd.Timedelta(hours=12))
    second = coverage.assign(lat_min=33., lat_max=43., lon_min=-126., lon_max=-116.,
                             start_utc=ISSUE - pd.Timedelta(hours=12))
    result = build(coverage=pd.concat([first, second]))
    assert result.fire_recent_complete.tolist() == [True, False]


@pytest.mark.parametrize("latitude,longitude", [(0., 179.9), (89.9, 0.), (-89.9, 0.)])
def test_dateline_and_pole_footprints_require_full_longitude_coverage(latitude, longitude):
    fires, coverage, _ = inputs()
    points = pd.DataFrame({"point_id": ["edge"], "lat": [latitude], "lon": [longitude]})
    world = build(fires, coverage, points)
    assert world.fire_recent_complete.all()
    partial = coverage.assign(lon_min=-179., lon_max=179.)
    assert not build(fires, partial, points).fire_recent_complete.any()


@pytest.mark.parametrize("column,value", [("lat_min", -91), ("lat_max", 91),
                                          ("lon_min", np.nan), ("lon_max", 181),
                                          ("lat_max", -90), ("lon_max", -180)])
def test_invalid_coverage_rectangles(column, value):
    _, coverage, _ = inputs()
    coverage[column] = value
    with pytest.raises(ValueError):
        build(coverage=coverage)


@pytest.mark.parametrize("seed", range(20))
def test_footprint_matches_independently_sampled_spherical_circle(seed):
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-60, 60), rng.uniform(-140, 140)
    radius = rng.uniform(100, 1000)
    lat_width, lon_width = rng.uniform(.5, 15, 2)
    fires, coverage, _ = inputs()
    coverage = coverage.assign(lat_min=lat - lat_width, lat_max=lat + lat_width,
                                lon_min=lon - lon_width, lon_max=lon + lon_width)
    points = pd.DataFrame({"point_id": ["probe"], "lat": [lat], "lon": [lon]})
    result = build(fires.iloc[:0], coverage, points, radius_km=radius)
    direction = np.linspace(0, 2 * np.pi, 10001)
    origin = np.radians(lat)
    angle = radius / 6371.
    circle_lat = np.arcsin(np.sin(origin) * np.cos(angle)
                            + np.cos(origin) * np.sin(angle) * np.cos(direction))
    circle_lon = np.radians(lon) + np.arctan2(
        np.sin(direction) * np.sin(angle) * np.cos(origin),
        np.cos(angle) - np.sin(origin) * np.sin(circle_lat))
    inside = ((np.degrees(circle_lat) >= lat - lat_width)
              & (np.degrees(circle_lat) <= lat + lat_width)
              & (np.degrees(circle_lon) >= lon - lon_width)
              & (np.degrees(circle_lon) <= lon + lon_width)).all()
    assert bool(result.fire_recent_complete.iloc[0]) == bool(inside)


def table_fixture():
    _, _, points = inputs()
    table = pd.concat([points.assign(issue_date="2023-07-03"),
                       points.iloc[::-1].assign(issue_date="2023-07-04")])
    table["y_L1"] = [12., np.nan, 200., 3.]
    table["obs_last1"] = [9., 3., 2., 15.]
    table["fire_dist_km"] = [100., 200., 300., 400.]
    return table


def test_augmentation_preserves_original_rows_index_targets_and_missingness():
    fires, coverage, _ = inputs()
    table = table_fixture()
    before = table.copy(deep=True)
    result = augment_fire_history(table, fires, coverage, source="VIIRS", issue_hour_utc=18.5)
    pd.testing.assert_frame_equal(result[table.columns], before)
    pd.testing.assert_frame_equal(table, before)
    np.testing.assert_allclose(result.fire_recent_frp, [30, 0, np.nan, np.nan])
    assert result.fire_recent_complete.tolist() == [True, True, False, False]
    assert "source" not in result
    assert "window_hours" not in result
    assert all(pd.api.types.is_numeric_dtype(result[c]) or result[c].dtype == bool
               for c in result.columns if c.startswith("fire_"))


@pytest.mark.parametrize("hour", [-1, 24, True, np.inf, np.nan])
def test_augmentation_rejects_invalid_issue_hour(hour):
    fires, coverage, _ = inputs()
    with pytest.raises(ValueError, match="issue_hour_utc"):
        augment_fire_history(table_fixture(), fires, coverage, source="VIIRS", issue_hour_utc=hour)


@pytest.mark.parametrize("problem", ["duplicate", "missing_id", "missing_date", "time", "overlap"])
def test_augmentation_rejects_ambiguous_or_existing_rows(problem):
    fires, coverage, _ = inputs()
    table = table_fixture().reset_index(drop=True)
    if problem == "duplicate":
        table = pd.concat([table, table.iloc[[0]]])
    elif problem == "missing_id":
        table.loc[0, "point_id"] = None
    elif problem == "missing_date":
        table.loc[0, "issue_date"] = None
    elif problem == "time":
        table.loc[0, "issue_date"] = "2023-07-03 12:00"
    else:
        table["fire_recent_count"] = 9
    with pytest.raises(ValueError):
        augment_fire_history(table, fires, coverage, source="VIIRS", issue_hour_utc=18.5)


def test_table_cli_matches_api(tmp_path):
    fires, coverage, _ = inputs()
    table = table_fixture()
    args = []
    for name, frame in (("fires", fires), ("coverage", coverage), ("table", table)):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path)
        args += [f"--{name}", str(path)]
    output = tmp_path / "augmented.parquet"
    args += ["--source", "VIIRS", "--issue-hour-utc", "18.5", "--out", str(output)]
    main(args)
    expected = augment_fire_history(table, fires, coverage, source="VIIRS", issue_hour_utc=18.5)
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)


def test_cli_round_trip_and_no_overwrite(tmp_path):
    fires, coverage, points = inputs()
    args = []
    for name, frame in (("fires", fires), ("coverage", coverage), ("points", points)):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path)
        args += [f"--{name}", str(path)]
    output = tmp_path / "new" / "features.parquet"
    args += ["--source", "VIIRS", "--issue-time", ISSUE.isoformat(), "--out", str(output)]
    main(args)
    pd.testing.assert_frame_equal(pd.read_parquet(output), build())
    metadata = json.loads(output.with_suffix(".parquet.json").read_text())
    assert metadata["source"] == "VIIRS"
    assert metadata["rows"] == len(points)
    assert metadata["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    for name in ("fires", "coverage", "points"):
        assert metadata["inputs"][name]["sha256"] == hashlib.sha256(
            (tmp_path / f"{name}.parquet").read_bytes()).hexdigest()
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before


def test_timestamp_metadata_has_stable_parquet_precision():
    result = build()
    for column in ("issue_time", "window_end_utc"):
        assert str(result[column].dtype) == "datetime64[ns, UTC]"


def test_existing_manifest_prevents_partial_output(tmp_path):
    output = tmp_path / "features.parquet"
    manifest = output.with_suffix(".parquet.json")
    manifest.write_text("existing evidence")
    with pytest.raises(SystemExit):
        main(["--fires", "missing", "--coverage", "missing", "--points", "missing",
              "--source", "VIIRS", "--issue-time", ISSUE.isoformat(), "--out", str(output)])
    assert not output.exists()
    assert manifest.read_text() == "existing evidence"


@pytest.mark.parametrize("competing_file", ["output", "manifest"])
def test_cli_never_overwrites_a_file_created_during_calculation(tmp_path, monkeypatch, competing_file):
    from serein.features import fire_history

    fires, coverage, points = inputs()
    args = []
    for name, frame in (("fires", fires), ("coverage", coverage), ("points", points)):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path)
        args += [f"--{name}", str(path)]
    output = tmp_path / "features.parquet"
    protected = output if competing_file == "output" else output.with_suffix(".parquet.json")
    result = build()

    def finish(*args, **kwargs):
        protected.write_bytes(b"concurrent result")
        return result

    monkeypatch.setattr(fire_history, "fire_history_features", finish)
    with pytest.raises(FileExistsError):
        main(args + ["--source", "VIIRS", "--issue-time", ISSUE.isoformat(), "--out", str(output)])
    assert protected.read_bytes() == b"concurrent result"


@pytest.mark.parametrize("seed", range(12))
def test_random_features_match_independent_scalar_oracle(seed):
    from math import asin, cos, radians, sin, sqrt

    rng = np.random.default_rng(seed)
    n = 73
    age = rng.uniform(-5, 55, n)
    lag = rng.uniform(0, 10, n)
    fires = pd.DataFrame({"latitude": rng.uniform(36, 41, n),
                          "longitude": rng.uniform(-124, -119, n), "source": "VIIRS",
                          "frp": rng.uniform(0, 300, n),
                          "time_utc": ISSUE - pd.to_timedelta(age, unit="h"),
                          "available_utc": ISSUE - pd.to_timedelta(age - lag, unit="h")})
    _, _, points = inputs()
    result = build(fires=fires)
    for i, point in enumerate(points.itertuples()):
        sums, counts = [], []
        for low, high in ((0, 24), (24, 48)):
            power = []
            for fire in fires.itertuples():
                elapsed = (ISSUE - fire.time_utc).total_seconds() / 3600
                a, b = radians(point.lat), radians(fire.latitude)
                dlon = radians(point.lon - fire.longitude)
                distance = 2 * 6371 * asin(sqrt(sin((a - b) / 2)**2
                                               + cos(a) * cos(b) * sin(dlon / 2)**2))
                if low <= elapsed < high and fire.available_utc <= ISSUE and distance <= 250:
                    power.append(fire.frp)
            sums.append(sum(power))
            counts.append(len(power))
        np.testing.assert_allclose(result.loc[i, ["fire_recent_frp", "fire_previous_frp"]]
                                   .to_numpy(dtype=float), sums)
        np.testing.assert_array_equal(result.loc[i, ["fire_recent_count", "fire_previous_count"]]
                                      .to_numpy(dtype=float), counts)
