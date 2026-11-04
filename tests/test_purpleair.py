from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from serein.data import purpleair


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
            return {"fields": ["time_stamp", "humidity"],
                    "data": [[self.params["start_timestamp"], 40.0]]}

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
    result = purpleair.download_history(1, start, end, max_days=1)
    assert result["time_stamp"].tolist() == [
        int(start.timestamp()), int(boundary.timestamp()), int(end.timestamp())]
    assert result["humidity"].tolist() == [40.0, 35.0, 50.0]
    pd.testing.assert_frame_equal(
        result, purpleair.download_history(1, start, end, max_days=1))


@pytest.mark.parametrize("kwargs", [
    {"max_days": 0}, {"max_days": -1}, {"average_minutes": 0}, {"average_minutes": -1},
])
def test_history_rejects_nonpositive_intervals(history_client, kwargs):
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError):
        purpleair.download_history(1, start, start + timedelta(days=1), **kwargs)
    assert not history_client


@pytest.mark.parametrize("delta", [timedelta(0), timedelta(days=-1)])
def test_history_rejects_empty_or_reversed_ranges(history_client, delta):
    start = datetime(2023, 7, 1)
    with pytest.raises(ValueError):
        purpleair.download_history(1, start, start + delta)
    assert not history_client
