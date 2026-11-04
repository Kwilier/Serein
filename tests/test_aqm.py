from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from serein.data import aqm, grib
from serein.data.grib import GribField


def test_extract_cycle_uses_current_grid_when_cell_count_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(aqm, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(aqm, "get", lambda *args, **kwargs: SimpleNamespace(content=b"cycle"))
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
        return SimpleNamespace(content=b"cycle")

    def decode(blob, with_grid=False):
        field = GribField("pm25", "surface", 0, 1, np.array([10.0, 20.0]), "ug/m3")
        return [field], ([0.0, 0.0], [0.0, 1.0])

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
    assert result["point_id"].tolist() == points["point_id"].tolist()
    expected = [20.0, 20.0] if change == "coordinates" else (
        [20.0, 10.0] if change == "order" else [10.0, 20.0])
    np.testing.assert_array_equal(result["value"], expected)
    loaded = aqm.load_cycles([day])
    pd.testing.assert_frame_equal(loaded.drop(columns="issue_date"), result)
