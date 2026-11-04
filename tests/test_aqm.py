from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from serein.data import aqm, grib
from serein.data.grib import GribField


@pytest.mark.parametrize("outcome", ["success", "empty", "read_error", "decode_error"])
def test_cycle_closes_response_before_decoding(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(aqm, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    failure = RuntimeError("download or decode failed")

    class Response:
        closed = False

        @property
        def content(self):
            if outcome == "read_error":
                raise failure
            return b"cycle"

        def close(self):
            self.closed = True

    response = Response()
    requests = []

    def fetch(url, **kwargs):
        requests.append(url)
        return None if len(requests) == 1 else response

    def decode(blob, with_grid=False):
        assert response.closed
        assert blob == b"cycle"
        if outcome == "decode_error":
            raise failure
        fields = [] if outcome == "empty" else [
            GribField("pm25", "surface", 0, 1, np.array([12.0]), "ug/m3")]
        return fields, ([0.0], [0.0])

    monkeypatch.setattr(aqm, "get", fetch)
    monkeypatch.setattr(aqm, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    if outcome.endswith("error"):
        with pytest.raises(RuntimeError) as exc:
            aqm.extract_cycle(date(2023, 8, 1), points)
        assert exc.value is failure
    else:
        result = aqm.extract_cycle(date(2023, 8, 1), points)
        if outcome == "empty":
            assert result is None
        else:
            assert result["value"].tolist() == [12.0]
    assert response.closed
    assert len(requests) == 2


def test_extract_cycle_uses_current_grid_when_cell_count_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(aqm, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(aqm, "get", lambda *args, **kwargs: SimpleNamespace(content=b"cycle", close=lambda: None))
    grids = [([0.0, 0.0], [0.0, 1.0]), ([0.0, 0.0], [1.0, 0.0])]
    values = [[10.0, 20.0], [20.0, 10.0]]
    calls = []

    def decode(blob, with_grid=False):
        i = len(calls)
        calls.append(with_grid)
        field = GribField("pm25", "surface", 0, 1, np.array(values[i]), "ug/m3")
        return [field], grids[i] if with_grid else None

    monkeypatch.setattr(aqm, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    first = aqm.extract_cycle(date(2023, 8, 1), points)
    second = aqm.extract_cycle(date(2023, 8, 2), points)

    np.testing.assert_array_equal(first["value"], [10.0])
    np.testing.assert_array_equal(second["value"], [10.0])


@pytest.mark.parametrize("change", ["points", "coordinates", "legacy", "order"])
def test_cycle_cache_matches_requested_points(tmp_path, monkeypatch, change):
    monkeypatch.setattr(aqm, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(content=b"cycle", close=lambda: None)

    def decode(blob, with_grid=False):
        fields = [GribField("pm25", "surface", h, h + 1,
                            np.array([10.0, 20.0]), "ug/m3") for h in range(72)]
        return fields, ([0.0, 0.0], [0.0, 1.0])

    monkeypatch.setattr(aqm, "get", fetch)
    monkeypatch.setattr(aqm, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A", "B"], "lat": [0.0, 0.0], "lon": [0.0, 1.0]})
    day = date(2023, 8, 1)
    first = aqm.extract_cycle(day, points)
    calls.clear()
    pd.testing.assert_frame_equal(aqm.extract_cycle(day, points), first)
    assert not calls
    if change == "points":
        points.loc[0, "point_id"] = "C"
    elif change == "coordinates":
        points.loc[0, "lon"] = 1.0
    elif change == "order":
        points = points.iloc[::-1]
    else:
        first.attrs.clear()
        first.to_parquet(tmp_path / "20230801_12z.parquet", index=False)
    result = aqm.extract_cycle(day, points)
    assert calls, "a changed point request must not reuse the cached extraction"
    assert result["point_id"].tolist() == points["point_id"].tolist() * 72
    expected = [20.0, 20.0] if change == "coordinates" else (
        [20.0, 10.0] if change == "order" else [10.0, 20.0])
    np.testing.assert_array_equal(result["value"], expected * 72)
    loaded = aqm.load_cycles([day])
    pd.testing.assert_frame_equal(loaded.drop(columns="issue_date"), result)


@pytest.mark.parametrize("day,hours", [(date(2021, 7, 19), 48),
                                     (date(2021, 7, 20), 72),
                                     (date(2024, 5, 14), 72)])
def test_partial_cycle_is_retried_until_all_hours_are_available(tmp_path, monkeypatch, day, hours):
    monkeypatch.setattr(aqm, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(content=b"cycle", close=lambda: None)

    def decode(blob, with_grid=False):
        steps = list(range(hours))
        if len(calls) == 1:
            steps[hours // 2] = 0
        fields = [GribField("pm25", "surface", h, h + 1,
                            np.array([float(h)]), "ug/m3") for h in steps]
        return fields, ([0.0], [0.0])

    monkeypatch.setattr(aqm, "get", fetch)
    monkeypatch.setattr(aqm, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    partial = aqm.extract_cycle(day, points)
    assert hours // 2 not in partial["step"].tolist()
    loaded = aqm.load_cycles([day])
    pd.testing.assert_frame_equal(loaded.drop(columns="issue_date"), partial)
    complete = aqm.extract_cycle(day, points)
    assert len(calls) == 2, "missing hours must trigger a fresh download"
    assert complete["step"].tolist() == list(range(hours))
    pd.testing.assert_frame_equal(aqm.extract_cycle(day, points), complete)
    assert len(calls) == 2, "complete cycles should reuse the cache"


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_nonfinite_in_grid_samples_are_retried(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(aqm, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(content=b"cycle", close=lambda: None)

    def decode(blob, with_grid=False):
        fields = [GribField("pm25", "surface", h, h + 1,
                            np.array([missing if h == 3 and len(calls) == 1 else 10.0]),
                            "ug/m3") for h in range(72)]
        return fields, ([0.0], [0.0])

    monkeypatch.setattr(aqm, "get", fetch)
    monkeypatch.setattr(aqm, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A", "outside"],
                           "lat": [0.0, 40.0], "lon": [0.0, 0.0]})
    day = date(2023, 8, 1)
    partial = aqm.extract_cycle(day, points)
    assert not np.isfinite(partial.loc[partial["step"] == 3, "value"]).any()
    loaded = aqm.load_cycles([day])
    pd.testing.assert_frame_equal(loaded.drop(columns="issue_date"), partial)
    repaired = aqm.extract_cycle(day, points)
    assert len(calls) == 2, "missing in-grid samples must trigger another download"
    assert (repaired.loc[repaired["point_id"] == "A", "value"] == 10.0).all()
    assert repaired.loc[repaired["point_id"] == "outside", "value"].isna().all()
    pd.testing.assert_frame_equal(aqm.extract_cycle(day, points), repaired)
    assert len(calls) == 2, "outside-grid points must not prevent cache reuse"
