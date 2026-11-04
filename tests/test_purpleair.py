from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from serein.data import purpleair


@pytest.mark.parametrize("column", ["pm2.5_cf_1_a", "pm2.5_cf_1_b", "humidity"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 20 + 1j, np.complex128(40)])
def test_clean_history_rejects_boolean_or_complex_measurements(column, value):
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=2, freq="h"),
        "sensor_index": [1, 1],
        "pm2.5_cf_1_a": [20.0, 20.0],
        "pm2.5_cf_1_b": [20.0, 20.0],
        "humidity": [40.0, 40.0],
    })
    frame[column] = pd.Series([20.0, value], dtype=object)
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="history measurement.*real numeric"):
        purpleair.clean_history(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", ["Float64", "Int64", object])
@pytest.mark.parametrize("column", ["pm2.5_cf_1_a", "pm2.5_cf_1_b", "humidity"])
def test_clean_history_accepts_nullable_measurements(dtype, column):
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=3, freq="h"),
        "sensor_index": [1, 1, 1],
        "pm2.5_cf_1_a": [20, 20, 300],
        "pm2.5_cf_1_b": [20, 20, 300],
        "humidity": [40, 40, 40],
    })
    frame[column] = pd.Series([40, pd.NA, pd.NA] if column == "humidity"
                              else [20, pd.NA, 300], dtype=dtype)
    original = frame.copy(deep=True)
    reference = frame.copy(deep=True)
    reference[column] = reference[column].to_numpy(dtype=float, na_value=np.nan)
    expected = purpleair.clean_history(reference)
    result = purpleair.clean_history(frame)
    pd.testing.assert_frame_equal(result, expected)
    assert len(result) == (3 if column == "humidity" else 2)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("kwargs", [
    {"n_sensors": -1}, {"days": -1}, {"average_minutes": 0},
    {"average_minutes": -10}, {"n_sensors": 1.5}, {"days": np.nan},
    {"average_minutes": np.inf}, {"n_sensors": True},
])
def test_cost_estimate_rejects_invalid_request_sizes(kwargs):
    request = {"n_sensors": 10, "days": 30, "average_minutes": 60}
    request.update(kwargs)
    with pytest.raises(ValueError):
        purpleair.estimate_history_points(**request)


@pytest.mark.parametrize("kwargs", [
    {"base_per_call": -1}, {"per_field_row": -1},
    {"base_per_call": np.nan}, {"per_field_row": np.inf},
    {"max_days_per_call_hourly": 0}, {"max_days_per_call_hourly": 1.5},
])
def test_cost_estimate_rejects_invalid_prices(kwargs):
    with pytest.raises(ValueError):
        purpleair.estimate_history_points(1, 1, prices=purpleair.PointPrices(**kwargs))


def test_clean_history_keeps_humidity_fallback_local_to_each_sensor():
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=7, freq="h"),
        "sensor_index": [1, 2, 1, 2, 3, 3, 1],
        "pm2.5_cf_1_a": [20.0] * 5 + [300.0, 20.0],
        "pm2.5_cf_1_b": [20.0] * 5 + [300.0, 20.0],
        "humidity": [20.0, 80.0, np.nan, np.inf, np.nan, np.nan, 40.0],
    }, index=[9, 7, 5, 3, 1, 8, 6])
    original = frame.copy(deep=True)
    result = purpleair.clean_history(frame)
    separate = pd.concat([
        purpleair.clean_history(group) for _, group in frame.groupby("sensor_index")
    ], ignore_index=True)
    pd.testing.assert_frame_equal(
        result.sort_values("time_utc").reset_index(drop=True),
        separate.sort_values("time_utc").reset_index(drop=True),
    )
    assert result["time_utc"].tolist() == frame.drop(index=1)["time_utc"].tolist()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("sensor", [None, np.nan, pd.NA])
def test_clean_history_rejects_missing_sensor_identifiers(sensor):
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=2, freq="h"),
        "sensor_index": [1, sensor],
        "pm2.5_cf_1_a": [20.0, 300.0],
        "pm2.5_cf_1_b": [20.0, 300.0],
        "humidity": [40.0, 40.0],
    })
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="sensor_index.*missing"):
        purpleair.clean_history(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("sensor", [0, -1, True, False, "1", "", 1.5, np.inf, -np.inf])
def test_clean_history_rejects_invalid_sensor_identifiers(sensor):
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=2, freq="h"),
        "sensor_index": pd.Series([1, sensor], dtype=object),
        "pm2.5_cf_1_a": [20.0, 300.0],
        "pm2.5_cf_1_b": [20.0, 300.0],
        "humidity": [40.0, 40.0],
    })
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="sensor_index.*positive integer"):
        purpleair.clean_history(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_clean_history_accepts_numpy_integer_sensor_identifiers():
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=2, freq="h"),
        "sensor_index": np.array([1, 2], dtype=np.int64),
        "pm2.5_cf_1_a": [10.0, 20.0],
        "pm2.5_cf_1_b": [10.0, 20.0],
        "humidity": [40.0, 80.0],
    })
    original = frame.copy(deep=True)
    result = purpleair.clean_history(frame)
    pd.testing.assert_series_equal(result["sensor_index"], frame["sensor_index"])
    np.testing.assert_allclose(result["pm25"], purpleair.epa_correction([10, 20], [40, 80]))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("stamp", [
    None, pd.NaT, "not a date", "2023-02-30", 1688169600, 1688169600.0,
    np.int64(1688169600), np.float64(1688169600.0),
])
def test_clean_history_rejects_invalid_observation_times(stamp):
    frame = pd.DataFrame({
        "time_utc": ["2023-07-01 00:00:00", stamp],
        "sensor_index": [1, 1],
        "pm2.5_cf_1_a": [10.0, 20.0],
        "pm2.5_cf_1_b": [10.0, 20.0],
        "humidity": [40.0, 40.0],
    })
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="history time_utc"):
        purpleair.clean_history(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("second", [
    "2023-07-01 00:00:00", "2023-06-30 17:00:00-07:00",
])
def test_clean_history_rejects_duplicate_sensor_observation_times(second):
    frame = pd.DataFrame({
        "time_utc": ["2023-07-01 00:00:00", second],
        "sensor_index": [1, 1],
        "pm2.5_cf_1_a": [10.0, 20.0],
        "pm2.5_cf_1_b": [10.0, 20.0],
        "humidity": [40.0, 80.0],
    }, index=[5, 5])
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match="duplicate.*sensor.*timestamp"):
        purpleair.clean_history(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_clean_history_accepts_simultaneous_observations_from_different_sensors():
    frame = pd.DataFrame({
        "time_utc": ["2023-07-01 00:00:00", "2023-06-30 17:00:00-07:00"],
        "sensor_index": [1, 2],
        "pm2.5_cf_1_a": [10.0, 20.0],
        "pm2.5_cf_1_b": [10.0, 20.0],
        "humidity": [40.0, 80.0],
    }, index=[5, 5])
    original = frame.copy(deep=True)
    result = purpleair.clean_history(frame)
    assert result["sensor_index"].tolist() == [1, 2]
    assert result["time_utc"].tolist() == frame["time_utc"].tolist()
    np.testing.assert_allclose(result["pm25"], purpleair.epa_correction([10, 20], [40, 80]))
    pd.testing.assert_frame_equal(frame, original)


def test_clean_history_preserves_valid_timezone_observation_times():
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=2, freq="h", tz="US/Pacific"),
        "sensor_index": [1, 1],
        "pm2.5_cf_1_a": [10.0, 20.0],
        "pm2.5_cf_1_b": [10.0, 20.0],
        "humidity": [40.0, 40.0],
    })
    original = frame.copy(deep=True)
    result = purpleair.clean_history(frame)
    pd.testing.assert_series_equal(result["time_utc"], frame["time_utc"])
    pd.testing.assert_frame_equal(frame, original)


def test_clean_history_preserves_row_alignment_with_duplicate_indices():
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=4, freq="h"),
        "sensor_index": [1, 2, 1, 2],
        "pm2.5_cf_1_a": [20.0, 20.0, 20.0, -1.0],
        "pm2.5_cf_1_b": [20.0, 20.0, 20.0, -1.0],
        "humidity": [20.0, 80.0, np.nan, np.nan],
    }, index=[5, 5, 5, 5])
    original = frame.copy(deep=True)
    result = purpleair.clean_history(frame)
    assert result["time_utc"].tolist() == frame["time_utc"].iloc[:3].tolist()
    assert result["sensor_index"].tolist() == [1, 2, 1]
    np.testing.assert_allclose(result["pm25"],
                               purpleair.epa_correction([20.0] * 3, [20.0, 80.0, 20.0]))
    pd.testing.assert_frame_equal(frame, original)


def test_cost_estimate_rounds_fractional_points_up():
    prices = purpleair.PointPrices(base_per_call=0.25, per_field_row=0.1)
    estimate = purpleair.estimate_history_points(1, 1, fields=["humidity"], prices=prices)
    assert estimate == {"sensors": 1, "days": 1, "calls": 1, "rows": 24, "points": 3}


@pytest.mark.parametrize("sensors,days", [(0, 30), (10, 0)])
def test_cost_estimate_empty_requests_cost_nothing(sensors, days):
    estimate = purpleair.estimate_history_points(sensors, days)
    assert estimate["calls"] == estimate["rows"] == estimate["points"] == 0


def test_correction_broadcasts_constant_humidity_across_pm_readings():
    pm = np.array([0.0, 20.0, 40.0, 100.0, 230.0, 300.0, np.nan])
    expected = purpleair.epa_correction(pm, np.full(pm.shape, 40.0))
    np.testing.assert_allclose(purpleair.epa_correction(pm, 40.0), expected)


def test_correction_broadcasts_constant_pm_across_humidity_readings():
    humidity = np.array([20.0, 40.0, 60.0])
    expected = purpleair.epa_correction(np.full(humidity.shape, 20.0), humidity)
    np.testing.assert_allclose(purpleair.epa_correction(20.0, humidity), expected)


@pytest.mark.parametrize("a,b", [(-1, -1), (-100, 10), (1, -1), (-1, 1)])
def test_clean_history_rejects_negative_channel_readings(a, b):
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=2, freq="h"),
        "sensor_index": [1, 1],
        "pm2.5_cf_1_a": [a, 0],
        "pm2.5_cf_1_b": [b, 0],
        "humidity": [40.0, 40.0],
    })
    result = purpleair.clean_history(frame)
    assert result["time_utc"].tolist() == [frame.loc[1, "time_utc"]]
    assert result["pm25"].notna().all()
    assert (result["pm25"] >= 0).all()


def test_clean_history_excludes_unavailable_corrections_without_mutating_input():
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=3, freq="h"),
        "sensor_index": [1, 1, 1],
        "pm2.5_cf_1_a": [20.0, 100.0, 300.0],
        "pm2.5_cf_1_b": [20.0, 100.0, 300.0],
        "humidity": [np.nan, np.nan, np.nan],
    }, index=[4, 6, 8])
    original = frame.copy(deep=True)
    result = purpleair.clean_history(frame)
    assert result["time_utc"].tolist() == [frame.loc[8, "time_utc"]]
    np.testing.assert_allclose(result["pm25"], purpleair.epa_correction([300.0], [np.nan]))
    assert result.index.tolist() == [0]
    pd.testing.assert_frame_equal(frame, original)


def test_clean_history_preserves_missing_humidity_median_fallback():
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=3, freq="h"),
        "sensor_index": [1, 1, 1],
        "pm2.5_cf_1_a": [20.0, 20.0, 20.0],
        "pm2.5_cf_1_b": [20.0, 20.0, 20.0],
        "humidity": [30.0, np.nan, 50.0],
    })
    result = purpleair.clean_history(frame)
    assert len(result) == len(frame)
    np.testing.assert_allclose(result["pm25"],
                               purpleair.epa_correction([20.0] * 3, [30.0, 40.0, 50.0]))


@pytest.mark.parametrize("invalid", [np.inf, -np.inf, -1.0, 101.0])
def test_clean_history_uses_only_valid_humidity_for_median_fallback(invalid):
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=4, freq="h"),
        "sensor_index": [1] * 4,
        "pm2.5_cf_1_a": [20.0] * 4,
        "pm2.5_cf_1_b": [20.0] * 4,
        "humidity": [30.0, invalid, np.nan, 50.0],
    }, index=[2, 4, 6, 8])
    original = frame.copy(deep=True)
    result = purpleair.clean_history(frame)
    assert result["time_utc"].tolist() == frame["time_utc"].tolist()
    np.testing.assert_allclose(result["pm25"],
                               purpleair.epa_correction([20.0] * 4, [30, 40, 40, 50]))
    pd.testing.assert_frame_equal(frame, original)


def test_clean_history_without_valid_humidity_preserves_humidity_independent_smoke():
    frame = pd.DataFrame({
        "time_utc": pd.date_range("2023-07-01", periods=4, freq="h"),
        "sensor_index": [1] * 4,
        "pm2.5_cf_1_a": [20.0, 100.0, 230.0, 300.0],
        "pm2.5_cf_1_b": [20.0, 100.0, 230.0, 300.0],
        "humidity": [np.inf, -np.inf, -1.0, 101.0],
    })
    result = purpleair.clean_history(frame)
    assert result["time_utc"].tolist() == [frame.loc[3, "time_utc"]]
    np.testing.assert_allclose(result["pm25"], purpleair.epa_correction([300.0], [np.nan]))


@pytest.fixture
def history_client(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(purpleair, "raw_dir", lambda source: tmp_path)
    monkeypatch.setattr(purpleair, "_headers", dict)
    monkeypatch.setattr(purpleair.time, "sleep", lambda seconds: None)

    class Response:
        def __init__(self, params):
            self.params = params

        def json(self):
            fields = self.params["fields"].split(",")
            return {"fields": ["time_stamp", *fields],
                    "data": [[self.params["start_timestamp"], *([40.0] * len(fields))]]}

    def get(url, *, headers, params):
        calls.append(params.copy())
        return Response(params)

    monkeypatch.setattr(purpleair, "get", get)
    return calls


def test_history_normalizes_offsets_and_reuses_equivalent_cache(history_client):
    start = datetime(2023, 7, 1, 12, tzinfo=timezone(timedelta(hours=-7)))
    end = start + timedelta(hours=2)
    first = purpleair.download_history(1, start, end)
    assert history_client[0]["start_timestamp"] == int(start.timestamp())
    assert history_client[0]["end_timestamp"] == int(end.timestamp())
    second = purpleair.download_history(1, start.astimezone(UTC), end.astimezone(UTC))
    assert len(history_client) == 1
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("sensor_index", [
    0, -1, 1.5, 1.0, True, np.bool_(True), None, np.nan, np.inf,
    "1", "../sensor", "1/history?fields=humidity",
])
def test_history_rejects_invalid_sensor_before_io(history_client, monkeypatch, sensor_index):
    def unexpected_cache_access(source):
        pytest.fail("invalid sensor reached the cache")

    monkeypatch.setattr(purpleair, "raw_dir", unexpected_cache_access)
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="sensor_index"):
        purpleair.download_history(sensor_index, start, start + timedelta(days=1))
    assert not history_client


@pytest.mark.parametrize("field", ["time_utc", "sensor_index"])
@pytest.mark.parametrize("empty", [False, True])
def test_history_rejects_response_metadata_collisions(
        history_client, monkeypatch, tmp_path, field, empty):
    class Response:
        def json(self):
            return {"fields": ["time_stamp", "humidity", field],
                    "data": [] if empty else [[1688169600, 40.0, 999]]}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="history response fields.*reserved"):
        purpleair.download_history(1, start, start + timedelta(days=1), fields=["humidity"])
    assert not list(tmp_path.rglob("*.parquet"))


def test_history_numpy_sensor_reuses_integer_cache(history_client):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=1)
    first = purpleair.download_history(np.int64(123), start, end)
    second = purpleair.download_history(123, start, end)
    assert len(history_client) == 1
    assert first["sensor_index"].tolist() == [123]
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("column", [
    "time_stamp", "time_utc", "sensor_index", *purpleair.HISTORY_FIELDS,
])
def test_history_rejects_cache_missing_columns(history_client, tmp_path, column):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=1)
    frame = purpleair.download_history(1, start, end)
    path = next(tmp_path.rglob("*.parquet"))
    frame.drop(columns=column).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="history cache missing columns"):
        purpleair.download_history(1, start, end)
    assert len(history_client) == 1
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame.drop(columns=column))


def test_history_cache_distinguishes_request_parameters(history_client):
    start = datetime(2023, 7, 1)
    end = start + timedelta(hours=2)
    purpleair.download_history(1, start, end)
    purpleair.download_history(1, start, end, average_minutes=10)
    purpleair.download_history(1, start, end, fields=["humidity"])
    purpleair.download_history(1, start + timedelta(hours=1), end)
    purpleair.download_history(2, start, end)
    assert len(history_client) == 5


def test_history_chunks_naive_times_as_utc(history_client):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=3, hours=1)
    purpleair.download_history(1, start, end, max_days=2)
    boundary = start + timedelta(days=2)
    assert [(p["start_timestamp"], p["end_timestamp"]) for p in history_client] == [
        (int(start.replace(tzinfo=UTC).timestamp()), int(boundary.replace(tzinfo=UTC).timestamp())),
        (int(boundary.replace(tzinfo=UTC).timestamp()), int(end.replace(tzinfo=UTC).timestamp())),
    ]


def test_history_merges_overlapping_chunks_in_time_order(history_client, monkeypatch):
    start = datetime(2023, 7, 1, tzinfo=UTC)
    boundary = start + timedelta(days=1)
    end = boundary + timedelta(days=1)
    responses = iter([
        {"fields": ["time_stamp", "humidity"], "data": [
            [int(boundary.timestamp()), 30.0], [int(start.timestamp()), 40.0]]},
        {"fields": ["time_stamp", "humidity"], "data": [
            [int(end.timestamp()), 50.0], [int(boundary.timestamp()), 35.0]]},
    ])

    class Response:
        def json(self):
            return next(responses)

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    result = purpleair.download_history(1, start, end, max_days=1, fields=["humidity"])
    assert result["time_stamp"].tolist() == [
        int(start.timestamp()), int(boundary.timestamp()), int(end.timestamp())]
    assert result["humidity"].tolist() == [40.0, 35.0, 50.0]
    pd.testing.assert_frame_equal(
        result, purpleair.download_history(1, start, end, max_days=1, fields=["humidity"]))


@pytest.mark.parametrize("missing", purpleair.HISTORY_FIELDS)
def test_history_rejects_missing_requested_measurements(
        history_client, monkeypatch, tmp_path, missing):
    class Response:
        def json(self):
            fields = ["time_stamp", *[f for f in purpleair.HISTORY_FIELDS if f != missing]]
            return {"fields": fields, "data": [[1688169600, 40.0, 40.0]]}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="missing requested fields"):
        purpleair.download_history(1, start, start + timedelta(days=1))
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("metadata", [
    {}, {"fields": None}, {"fields": "time_stamp,humidity"},
    {"fields": {"time_stamp": 0, "humidity": 1}},
    {"fields": ["time_stamp", None]}, {"fields": ["time_stamp", ["humidity"]]},
    {"fields": ["time_stamp", ""]}, {"fields": ["time_stamp", 1]},
])
@pytest.mark.parametrize("rows", [[], [[1688169600, 40.0]]])
def test_history_rejects_malformed_field_metadata(
        history_client, monkeypatch, tmp_path, metadata, rows):
    class Response:
        def json(self):
            return {**metadata, "data": rows}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="history response fields"):
        purpleair.download_history(1, start, start + timedelta(days=1), fields=["humidity"])
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("stamp", [None, np.nan, np.inf, "invalid", True, 1688169600.5])
def test_history_rejects_invalid_response_timestamps(history_client, monkeypatch, tmp_path, stamp):
    class Response:
        def json(self):
            return {"fields": ["time_stamp", "humidity"], "data": [[stamp, 40.0]]}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="time_stamp"):
        purpleair.download_history(1, start, start + timedelta(days=1))
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("offset", [-1, 86401])
def test_history_rejects_response_outside_chunk(history_client, monkeypatch, tmp_path, offset):
    start = datetime(2023, 7, 1, tzinfo=UTC)

    class Response:
        def json(self):
            return {"fields": ["time_stamp", "humidity"],
                    "data": [[int(start.timestamp()) + offset, 40.0]]}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="requested chunk"):
        purpleair.download_history(1, start, start + timedelta(days=2), max_days=1)
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("fields,row", [
    (["time_stamp", "time_stamp"], [1688169600, 1688169600]),
    (["time_stamp", "humidity", "humidity"], [1688169600, 40.0, 50.0]),
])
def test_history_rejects_duplicate_response_fields(
        history_client, monkeypatch, tmp_path, fields, row):
    class Response:
        def json(self):
            return {"fields": fields, "data": [row]}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="unique"):
        purpleair.download_history(1, start, start + timedelta(days=1))
    assert not list(tmp_path.rglob("*.parquet"))


def test_history_rejects_missing_response_timestamp(history_client, monkeypatch, tmp_path):
    class Response:
        def json(self):
            return {"fields": ["humidity"], "data": [[40.0]]}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="time_stamp"):
        purpleair.download_history(1, start, start + timedelta(days=1))
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("rows", [
    [[1688169600]],
    [[1688169600, 40.0], [1688173200]],
    [{"time_stamp": 1688169600, "humidity": 40.0}],
])
def test_history_rejects_malformed_response_rows(history_client, monkeypatch, tmp_path, rows):
    class Response:
        def json(self):
            return {"fields": ["time_stamp", "humidity"], "data": rows}

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="response rows"):
        purpleair.download_history(1, start, start + timedelta(days=1))
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("kwargs", [
    {"max_days": 0}, {"max_days": -1}, {"average_minutes": 0}, {"average_minutes": -1},
])
def test_history_rejects_nonpositive_intervals(history_client, kwargs):
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError):
        purpleair.download_history(1, start, start + timedelta(days=1), **kwargs)
    assert not history_client


@pytest.mark.parametrize("payload", [
    {}, {"data": None}, {"data": {}}, {"data": ""}, {"data": False},
    {"data": 0}, [], None,
])
def test_history_rejects_malformed_empty_chunk(history_client, monkeypatch, tmp_path, payload):
    responses = iter([
        {"fields": ["time_stamp", "humidity"], "data": [[1688169600, 40.0]]},
        payload,
    ])

    class Response:
        def json(self):
            return next(responses)

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="history response"):
        purpleair.download_history(1, start, start + timedelta(days=2),
                                  max_days=1, fields=["humidity"])
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("fields,match", [
    (["time_stamp", "humidity", "humidity"], "unique"),
    (["humidity"], "time_stamp"),
    (["time_stamp"], "missing requested fields"),
    ([], "time_stamp"),
])
def test_history_rejects_incomplete_empty_chunk_fields(
        history_client, monkeypatch, tmp_path, fields, match):
    responses = iter([
        {"fields": ["time_stamp", "humidity"], "data": [[1688169600, 40.0]]},
        {"fields": fields, "data": []},
    ])

    class Response:
        def json(self):
            return next(responses)

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match=match):
        purpleair.download_history(1, start, start + timedelta(days=2),
                                  max_days=1, fields=["humidity"])
    assert not list(tmp_path.rglob("*.parquet"))


def test_history_accepts_empty_chunk_between_populated_chunks(history_client, monkeypatch):
    responses = iter([
        {"fields": ["time_stamp", "humidity"], "data": [[1688169600, 40.0]]},
        {"fields": ["time_stamp", "humidity"], "data": []},
        {"fields": ["time_stamp", "humidity"], "data": [[1688342400, 50.0]]},
    ])

    class Response:
        def json(self):
            return next(responses)

    monkeypatch.setattr(purpleair, "get", lambda *args, **kwargs: Response())
    start = datetime(2023, 7, 1)
    result = purpleair.download_history(1, start, start + timedelta(days=3),
                                        max_days=1, fields=["humidity"])
    assert result["time_stamp"].tolist() == [1688169600, 1688342400]
    assert result["humidity"].tolist() == [40.0, 50.0]


@pytest.mark.parametrize("name", ["max_days", "average_minutes"])
@pytest.mark.parametrize("value", [1.5, np.nan, np.inf, -np.inf, True])
def test_history_rejects_noninteger_intervals(history_client, name, value):
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError):
        purpleair.download_history(1, start, start + timedelta(days=1), **{name: value})
    assert not history_client


@pytest.mark.parametrize("pause", [-1, np.nan, np.inf, -np.inf])
def test_history_rejects_invalid_pause_before_request(history_client, pause):
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError, match="pause_s"):
        purpleair.download_history(1, start, start + timedelta(days=1), pause_s=pause)
    assert not history_client


def test_history_accepts_numpy_integer_intervals_and_zero_pause(history_client):
    start = datetime(2023, 7, 1)
    purpleair.download_history(1, start, start + timedelta(days=2),
                              max_days=np.int64(1), average_minutes=np.int64(60), pause_s=0)
    assert len(history_client) == 2
    assert all(call["average"] == 60 for call in history_client)


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(days=-1)])
def test_history_rejects_empty_or_reversed_ranges(history_client, delta):
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError):
        purpleair.download_history(1, start, start + delta)
    assert not history_client


@pytest.mark.parametrize("start_us,end_us", [(1, 0), (0, 1), (100, 200)])
def test_history_rejects_subsecond_boundaries_before_cache_access(
        history_client, monkeypatch, start_us, end_us):
    def unexpected_cache_access(source):
        pytest.fail("invalid range reached the cache")

    monkeypatch.setattr(purpleair, "raw_dir", unexpected_cache_access)
    start = datetime(2023, 7, 1, microsecond=start_us)
    end = datetime(2023, 7, 1, microsecond=end_us)
    if end <= start:
        end += timedelta(seconds=1)
    with pytest.raises(ValueError, match="whole seconds"):
        purpleair.download_history(1, start, end)
    assert not history_client


@pytest.mark.parametrize("fields", [
    None, "humidity", [], [""], [" "], [None], [1],
    ["humidity", "humidity"], ["humidity,temperature"], [" humidity"],
])
@pytest.mark.parametrize("operation", ["estimate", "download"])
def test_history_rejects_invalid_request_fields_before_io(
        history_client, monkeypatch, fields, operation):
    def unexpected_cache_access(source):
        pytest.fail("invalid fields reached the cache")

    monkeypatch.setattr(purpleair, "raw_dir", unexpected_cache_access)
    with pytest.raises(ValueError, match="fields"):
        if operation == "estimate":
            purpleair.estimate_history_points(1, 1, fields=fields)
        else:
            start = datetime(2023, 7, 1)
            purpleair.download_history(1, start, start + timedelta(days=1), fields=fields)
    assert not history_client


@pytest.mark.parametrize("sensor", [2, 0, -1, None, np.nan, True, "1", 1.5])
def test_history_rejects_cache_for_invalid_or_different_sensor(history_client, tmp_path, sensor):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=1)
    frame = purpleair.download_history(1, start, end)
    path = next(tmp_path.rglob("*.parquet"))
    frame["sensor_index"] = sensor
    frame.to_parquet(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="history cache sensor_index"):
        purpleair.download_history(1, start, end)
    assert len(history_client) == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("stamp", [None, np.nan, np.inf, True, "1688169600",
                                   1688169600.5, 1688169599, 1688256001])
def test_history_rejects_invalid_cached_timestamp(history_client, tmp_path, stamp):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=1)
    frame = purpleair.download_history(1, start, end)
    path = next(tmp_path.rglob("*.parquet"))
    frame["time_stamp"] = stamp
    frame.to_parquet(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="history cache time_stamp"):
        purpleair.download_history(1, start, end)
    assert len(history_client) == 1
    assert path.read_bytes() == original


@pytest.mark.parametrize("utc", [None, "not a date", "2023-07-01 01:00:00"])
def test_history_rejects_inconsistent_cached_utc(history_client, tmp_path, utc):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=1)
    frame = purpleair.download_history(1, start, end)
    path = next(tmp_path.rglob("*.parquet"))
    frame["time_utc"] = utc
    frame.to_parquet(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="history cache time_utc"):
        purpleair.download_history(1, start, end)
    assert len(history_client) == 1
    assert path.read_bytes() == original


def test_history_accepts_cached_utc_with_equivalent_timezone(history_client, tmp_path):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=1)
    frame = purpleair.download_history(1, start, end)
    path = next(tmp_path.rglob("*.parquet"))
    frame["time_stamp"] = frame["time_stamp"].astype(float)
    frame["time_utc"] = frame["time_utc"].dt.tz_localize("UTC").dt.tz_convert("US/Pacific")
    frame.to_parquet(path, index=False)
    result = purpleair.download_history(1, start, end)
    pd.testing.assert_frame_equal(result, frame)
    assert len(history_client) == 1


@pytest.mark.parametrize("order", [[0, 0, 1], [0, 1, 0], [1, 0]])
def test_history_rejects_duplicate_or_unsorted_cached_rows(history_client, tmp_path, order):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=2)
    frame = purpleair.download_history(1, start, end, max_days=1)
    path = next(tmp_path.rglob("*.parquet"))
    frame.iloc[order].to_parquet(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="history cache time_stamp.*unique.*sorted"):
        purpleair.download_history(1, start, end, max_days=1)
    assert len(history_client) == 2
    assert path.read_bytes() == original


def test_history_reuses_sorted_multirow_cache(history_client):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=3)
    first = purpleair.download_history(1, start, end, max_days=1)
    second = purpleair.download_history(1, start, end, max_days=1)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3
    assert len(history_client) == 3


@pytest.mark.parametrize("field", ["time_utc", "sensor_index"])
@pytest.mark.parametrize("operation", ["estimate", "download"])
def test_history_rejects_generated_metadata_fields_before_io(
        history_client, monkeypatch, field, operation):
    def unexpected_cache_access(source):
        pytest.fail("reserved fields reached the cache")

    monkeypatch.setattr(purpleair, "raw_dir", unexpected_cache_access)
    with pytest.raises(ValueError, match="fields.*reserved"):
        if operation == "estimate":
            purpleair.estimate_history_points(1, 1, fields=["humidity", field])
        else:
            start = datetime(2023, 7, 1)
            purpleair.download_history(1, start, start + timedelta(days=1),
                                      fields=["humidity", field])
    assert not history_client
