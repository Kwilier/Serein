from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from serein.data import purpleair


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


def test_history_numpy_sensor_reuses_integer_cache(history_client):
    start = datetime(2023, 7, 1)
    end = start + timedelta(days=1)
    first = purpleair.download_history(np.int64(123), start, end)
    second = purpleair.download_history(123, start, end)
    assert len(history_client) == 1
    assert first["sensor_index"].tolist() == [123]
    pd.testing.assert_frame_equal(first, second)


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
