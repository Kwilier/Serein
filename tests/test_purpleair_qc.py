import json

import numpy as np
import pandas as pd
import pytest

from serein.cli import main
from serein.data.purpleair import epa_correction
from serein.data.purpleair_qc import (
    QCConfig,
    accepted_history,
    quality_control,
    summarize_qc,
)


def history(n=4):
    return pd.DataFrame({
        "sensor_index": np.arange(1, n + 1),
        "time_utc": ["2023-08-01T12:00:00Z"] * n,
        "pm2.5_cf_1_a": np.arange(n, dtype=float) + 10,
        "pm2.5_cf_1_b": np.arange(n, dtype=float) + 11,
        "humidity": 40.0,
        "lat": 38.58 + np.arange(n) * 0.001,
        "lon": -121.49,
        "location_type": 0,
    })


def test_valid_rows_retain_original_values_order_index_and_corrected_pm():
    source = history()
    source.index = [7, 7, 3, 2]
    original = source.copy(deep=True)
    result = quality_control(source)

    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(result[source.columns], source)
    assert result["qc_keep"].all()
    assert result["qc_neighbor_evaluated"].all()
    assert result["qc_flags"].eq("").all()
    np.testing.assert_allclose(result.pm25, epa_correction(np.arange(4) + 10.5, 40))
    assert accepted_history(result).columns.tolist() == ["time_utc", "sensor_index", "pm25"]
    assert summarize_qc(result)["kept"] == 4


@pytest.mark.parametrize("column,value,flag", [
    ("pm2.5_cf_1_a", np.nan, "invalid_channels"),
    ("pm2.5_cf_1_b", np.inf, "invalid_channels"),
    ("pm2.5_cf_1_a", -1.0, "invalid_channels"),
    ("pm2.5_cf_1_a", "missing", "invalid_channels"),
    ("pm2.5_cf_1_b", 100.0, "ab_disagreement"),
    ("humidity", np.nan, "invalid_humidity"),
    ("humidity", -1.0, "invalid_humidity"),
    ("humidity", 101.0, "invalid_humidity"),
    ("sensor_index", 0, "invalid_sensor"),
    ("sensor_index", 1.5, "invalid_sensor"),
    ("sensor_index", True, "invalid_sensor"),
    ("time_utc", None, "invalid_time"),
    ("time_utc", "not a time", "invalid_time"),
    ("time_utc", 1690891200, "invalid_time"),
    ("lat", 91, "missing_coordinates"),
    ("lon", np.nan, "missing_coordinates"),
    ("location_type", "unknown", "unknown_location"),
])
def test_bad_rows_have_reasons_and_never_get_corrected_pm(column, value, flag):
    frame = history(1)
    frame[column] = value
    result = quality_control(frame)
    assert result.loc[0, f"qc_{flag}"]
    assert flag in result.loc[0, "qc_flags"].split(";")
    assert not result.loc[0, "qc_keep"]
    assert np.isnan(result.loc[0, "pm25"])


@pytest.mark.parametrize("column,flag", [
    ("pm2.5_cf_1_a", "invalid_channels"), ("lat", "missing_coordinates"),
    ("location_type", "unknown_location"),
])
def test_unavailable_fields_are_flagged_instead_of_silently_clean(column, flag):
    result = quality_control(history(1).drop(columns=column))
    assert result.loc[0, f"qc_{flag}"]
    assert not result.loc[0, "qc_keep"]


@pytest.mark.parametrize("value", [1, "1", "indoor", " INDOOR "])
def test_indoor_location_type_rows_are_excluded(value):
    frame = history(1)
    frame["location_type"] = value
    result = quality_control(frame)
    assert result.loc[0, "qc_indoor"]
    assert not result.loc[0, "qc_keep"]


@pytest.mark.parametrize("indoor", [True, "true", "yes", "1"])
def test_indoor_boolean_and_string_metadata_is_understood(indoor):
    frame = history(1).drop(columns="location_type")
    frame["indoor"] = indoor
    result = quality_control(frame)
    assert result.loc[0, "qc_indoor"]
    assert not result.loc[0, "qc_keep"]


@pytest.mark.parametrize("indoor", [False, "false", "no", "0"])
def test_outdoor_boolean_and_string_metadata_is_understood(indoor):
    frame = history(1).drop(columns="location_type")
    frame["indoor"] = indoor
    assert quality_control(frame).loc[0, "qc_keep"]


def test_conflicting_location_fields_are_not_accepted():
    frame = history(1)
    frame["indoor"] = True
    result = quality_control(frame)
    assert result.loc[0, "qc_unknown_location"]
    assert not result.loc[0, "qc_keep"]


def test_metadata_can_supply_coordinates_and_indoor_status_without_mutation():
    source = history().drop(columns=["lat", "lon", "location_type"])
    sensors = history()[["sensor_index", "lat", "lon", "location_type"]]
    sensors = sensors.rename(columns={"lat": "latitude", "lon": "longitude"})
    original = sensors.copy(deep=True)
    result = quality_control(source, sensors)
    assert result["qc_keep"].all()
    pd.testing.assert_frame_equal(sensors, original)
    assert set(source.columns) < set(result.columns)
    assert "lat" not in source


def test_duplicate_metadata_rejected_and_unknown_sensor_flagged():
    sensors = history(1)[["sensor_index", "lat", "lon", "location_type"]]
    with pytest.raises(ValueError, match="unique positive integer"):
        quality_control(history(1), pd.concat([sensors, sensors]))
    frame = history(1).drop(columns=["lat", "lon", "location_type"])
    frame["sensor_index"] = 999
    result = quality_control(frame, sensors)
    assert not result.loc[0, "qc_keep"]
    assert result.loc[0, "qc_missing_coordinates"]
    assert result.loc[0, "qc_unknown_location"]


def test_stuck_runs_require_both_channels_and_consecutive_timestamps():
    frame = pd.concat([history(1)] * 9, ignore_index=True)
    frame["time_utc"] = pd.date_range("2023-08-01", periods=9, freq="h")
    frame["pm2.5_cf_1_a"] = [10, 10, 10, 20, 20, 20, 30, 30, 30]
    frame["pm2.5_cf_1_b"] = [11, 11, 11, 21, 21, 21, 31, 32, 33]
    # A missing hour splits the middle plateau into two shorter runs.
    frame.loc[5, "time_utc"] += pd.Timedelta(minutes=30)
    result = quality_control(frame, config=QCConfig(stuck_hours=3))
    assert result["qc_stuck"].tolist() == [False, False, True, False, False, False,
                                            False, False, False]
    assert pd.isna(result.loc[2, "pm25"])


def test_invalid_reading_breaks_a_stuck_run_and_input_order_is_preserved():
    frame = pd.concat([history(1)] * 5, ignore_index=True)
    frame["time_utc"] = pd.date_range("2023-08-01", periods=5, freq="h")
    frame.loc[2, "pm2.5_cf_1_a"] = np.nan
    frame = frame.iloc[[4, 2, 0, 3, 1]].reset_index(drop=True)
    result = quality_control(frame, config=QCConfig(stuck_hours=3))
    assert not result["qc_stuck"].any()
    assert result["time_utc"].tolist() == frame["time_utc"].tolist()


def test_slowly_drifting_channels_are_not_marked_as_flat():
    frame = pd.concat([history(1)] * 5, ignore_index=True)
    frame["time_utc"] = pd.date_range("2023-08-01", periods=5, freq="h")
    frame["pm2.5_cf_1_a"] = 10 + np.arange(5) * 0.009
    frame["pm2.5_cf_1_b"] = 11 + np.arange(5) * 0.009
    result = quality_control(frame, config=QCConfig(stuck_hours=3, stuck_tolerance=0.01))
    assert not result["qc_stuck"].any()


def test_duplicate_instants_in_different_offsets_are_rejected():
    frame = pd.concat([history(1)] * 2, ignore_index=True)
    frame["time_utc"] = ["2023-08-01T12:00:00Z", "2023-08-01T05:00:00-07:00"]
    result = quality_control(frame)
    assert result["qc_duplicate"].all()
    assert not result["qc_keep"].any()


def test_epoch_timestamp_fallback_is_normalized_to_utc():
    frame = history(1).drop(columns="time_utc")
    frame["time_stamp"] = int(pd.Timestamp("2023-08-01T12:00:00Z").timestamp())
    result = quality_control(frame)
    assert result.loc[0, "qc_keep"]
    assert result.loc[0, "time_utc_normalized"] == pd.Timestamp("2023-08-01T12:00:00")


@pytest.mark.parametrize("stamp", [True, False])
def test_boolean_epoch_stamps_are_rejected(stamp):
    frame = history(1).drop(columns="time_utc")
    frame["time_stamp"] = stamp
    result = quality_control(frame)
    assert result.loc[0, "qc_invalid_time"]
    assert not result.loc[0, "qc_keep"]


def test_isolated_high_reading_is_flagged_when_enough_peers_are_available():
    frame = history(5)
    frame.loc[4, ["pm2.5_cf_1_a", "pm2.5_cf_1_b"]] = 300.0
    result = quality_control(frame)
    assert result["qc_neighbor_evaluated"].all()
    assert result["qc_neighbor_outlier"].tolist() == [False] * 4 + [True]
    assert pd.isna(result.loc[4, "pm25"])


def test_widespread_high_smoke_is_not_treated_as_a_neighbor_outlier():
    frame = history(5)
    frame["pm2.5_cf_1_a"] = [280, 300, 320, 340, 360]
    frame["pm2.5_cf_1_b"] = frame["pm2.5_cf_1_a"] + 2
    result = quality_control(frame)
    assert result["qc_neighbor_evaluated"].all()
    assert not result["qc_neighbor_outlier"].any()
    assert result["qc_keep"].all()


def test_neighbor_test_reports_insufficient_peers_and_never_counts_a_sensor_twice():
    frame = pd.concat([history(3), history(1)], ignore_index=True)
    frame.loc[3, "time_utc"] = "2023-08-01T12:30:00Z"
    result = quality_control(frame)
    assert not result["qc_neighbor_evaluated"].any()
    assert not result["qc_neighbor_outlier"].any()
    assert result["qc_keep"].all()


def test_distant_sensors_do_not_count_as_neighbors():
    frame = history(4)
    frame["lat"] = [30, 35, 40, 45]
    result = quality_control(frame)
    assert not result["qc_neighbor_evaluated"].any()


def test_later_readings_never_change_earlier_stuck_or_neighbor_flags():
    frame = pd.concat([history(1)] * 5, ignore_index=True)
    frame["time_utc"] = pd.date_range("2023-08-01", periods=5, freq="h")
    config = QCConfig(stuck_hours=3)
    before = quality_control(frame.iloc[:2], config=config)
    after = quality_control(frame, config=config)
    assert after["qc_stuck"].tolist() == [False, False, True, True, True]
    pd.testing.assert_frame_equal(before, after.iloc[:2])

    frame = history(4)
    frame.loc[1:, "time_utc"] = "2023-08-01T12:30:00Z"
    frame.loc[0, ["pm2.5_cf_1_a", "pm2.5_cf_1_b"]] = 300.0
    result = quality_control(frame)
    assert not result.loc[0, "qc_neighbor_evaluated"]
    assert not result.loc[0, "qc_neighbor_outlier"]


@pytest.mark.parametrize("kwargs", [
    {"min_neighbors": 0}, {"min_neighbors": True}, {"expected_minutes": 1.5},
    {"stuck_hours": 0}, {"stuck_tolerance": -1}, {"neighbor_radius_km": np.inf},
    {"neighbor_radius_km": 0}, {"neighbor_rel_tol": np.nan},
])
def test_invalid_configuration_fails_clearly(kwargs):
    with pytest.raises((TypeError, ValueError)):
        QCConfig(**kwargs)


def test_empty_input_and_extreme_measurements_are_handled_without_infinite_output():
    empty = quality_control(pd.DataFrame())
    assert empty.empty
    assert accepted_history(empty).columns.tolist() == ["time_utc", "sensor_index", "pm25"]
    frame = history(2)
    frame.loc[:, ["pm2.5_cf_1_a", "pm2.5_cf_1_b"]] = [[1e308, 1e308], [1e200, 1e200]]
    result = quality_control(frame)
    assert result["qc_invalid_channels"].all()
    assert not result["qc_keep"].any()
    assert result["pm25"].isna().all()


def test_cli_writes_audit_and_clean_output_without_network(tmp_path, capsys, monkeypatch):
    from serein.data import purpleair

    def unexpected(*args, **kwargs):
        pytest.fail("offline QC must not make API requests")

    monkeypatch.setattr(purpleair, "get", unexpected)
    source, audit_path, clean_path = (tmp_path / name for name in
                                     ("history.csv", "audit.parquet", "clean.csv"))
    frame = history()
    frame.loc[0, "location_type"] = 1
    frame.to_csv(source, index=False)
    main(["purpleair", "qc", str(source), "--output", str(audit_path),
          "--clean-output", str(clean_path)])
    audit = pd.read_parquet(audit_path)
    clean = pd.read_csv(clean_path)
    summary = json.loads(capsys.readouterr().out)
    assert len(audit) == 4 and len(clean) == 3
    assert summary["kept"] == 3 and summary["flags"]["indoor"] == 1
    assert clean["pm25"].notna().all()
    pd.testing.assert_frame_equal(pd.read_csv(source), frame)


def test_cli_rejects_overwriting_source_or_invalid_destinations_before_writing(tmp_path):
    source, target = tmp_path / "history.csv", tmp_path / "audit.csv"
    history().to_csv(source, index=False)
    with pytest.raises(SystemExit, match="distinct"):
        main(["purpleair", "qc", str(source), "--output", str(source)])
    with pytest.raises(SystemExit, match=".csv or .parquet"):
        main(["purpleair", "qc", str(source), "--output", str(target),
              "--clean-output", str(tmp_path / "bad.txt")])
    assert not target.exists()


def test_cli_requires_input_and_audit_destination():
    with pytest.raises(SystemExit, match="requires INPUT and --output"):
        main(["purpleair", "qc"])
