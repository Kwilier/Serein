from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.data import grib, hrrr
from serein.data.grib import GribField


@pytest.mark.parametrize("missing", ["hour", "field", "decoded"])
def test_partial_cycle_is_retried_until_complete(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(hrrr, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    partial = True
    calls = []

    def fetch(d, cycle, fhr, wanted):
        calls.append(fhr)
        if partial and missing == "hour" and fhr == 1:
            return None
        return [(name, b"empty" if partial and missing == "decoded" and name != hrrr.SMOKE
                 else b"field") for name in wanted
                if not (partial and missing == "field" and name != hrrr.SMOKE)]

    def decode(blob, with_grid=False):
        fields = [] if blob == b"empty" else [GribField(
            "smoke", "surface", 0, 0, np.array([10.0]), "ug/m3")]
        return fields, ([0.0], [0.0]) if with_grid else None

    monkeypatch.setattr(hrrr, "_fetch_fields", fetch)
    monkeypatch.setattr(hrrr, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    kwargs = {"max_fhr": 1, "met_every": 1,
              "met_vars": [hrrr.DEFAULT_MET[0]], "workers": 1}
    day = date(2023, 8, 1)
    first = hrrr.extract_cycle(day, points, **kwargs)
    assert len(first) < 4
    assert len(hrrr.load_cycles([day])) == len(first)
    partial = False
    calls.clear()
    complete = hrrr.extract_cycle(day, points, **kwargs)
    assert calls, "missing hours or fields must be retried"
    assert len(complete) == 4
    calls.clear()
    pd.testing.assert_frame_equal(hrrr.extract_cycle(day, points, **kwargs), complete)
    assert not calls


@pytest.mark.parametrize("setting,value", [
    ("max_fhr", -1), ("smoke_every", 0), ("smoke_every", -1),
    ("met_every", 0), ("met_every", -1), ("workers", 0),
])
def test_extract_cycle_rejects_invalid_settings_before_io(monkeypatch, setting, value):
    def unexpected_io(*args, **kwargs):
        pytest.fail("invalid settings must be rejected before cache or network access")

    monkeypatch.setattr(hrrr, "interim_dir", unexpected_io)
    monkeypatch.setattr(hrrr, "_fetch_fields", unexpected_io)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    with pytest.raises(ValueError, match=setting):
        hrrr.extract_cycle(date(2023, 8, 1), points, **{setting: value})


@pytest.mark.parametrize("met_vars", [None, [], [hrrr.DEFAULT_MET[0]]])
def test_extract_cycle_respects_weather_field_selection(tmp_path, monkeypatch, met_vars):
    monkeypatch.setattr(hrrr, "interim_dir", lambda name: tmp_path)
    requests = []

    def fetch(d, cycle, fhr, wanted):
        requests.append((fhr, wanted.copy()))

    monkeypatch.setattr(hrrr, "_fetch_fields", fetch)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    hrrr.extract_cycle(date(2023, 8, 1), points, max_fhr=3, smoke_every=2,
                       met_every=3, met_vars=met_vars, workers=1)

    weather = hrrr.DEFAULT_MET if met_vars is None else met_vars
    expected = [(0, [hrrr.SMOKE] + weather), (2, [hrrr.SMOKE])]
    if weather:
        expected.append((3, weather))
    assert requests == expected


def test_extract_cycle_uses_current_grid_when_cell_count_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(hrrr, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(hrrr, "_fetch_fields", lambda *args: [(hrrr.SMOKE, b"cycle")])
    grids = [([0.0, 0.0], [0.0, 1.0]), ([0.0, 0.0], [1.0, 0.0])]
    values = [[10.0, 20.0], [20.0, 10.0]]
    calls = []

    def decode(blob, with_grid=False):
        i = len(calls)
        calls.append(with_grid)
        field = GribField("smoke", "surface", 0, 0, np.array(values[i]), "ug/m3")
        return [field], grids[i] if with_grid else None

    monkeypatch.setattr(hrrr, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    first = hrrr.extract_cycle(date(2023, 8, 1), points, max_fhr=0, workers=1)
    second = hrrr.extract_cycle(date(2023, 8, 2), points, max_fhr=0, workers=1)

    np.testing.assert_array_equal(first["value"], [10.0])
    np.testing.assert_array_equal(second["value"], [10.0])


@pytest.mark.parametrize("change", ["points", "coordinates", "legacy", "fields", "horizon",
                                  "smoke_interval", "met_interval"])
def test_cycle_cache_matches_requested_points_and_settings(tmp_path, monkeypatch, change):
    monkeypatch.setattr(hrrr, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    calls = []

    def fetch(d, cycle, fhr, wanted):
        calls.append((fhr, wanted.copy()))
        return [(name, b"field") for name in wanted]

    def decode(blob, with_grid=False):
        field = GribField("smoke", "surface", 0, 0, np.array([10.0, 20.0]), "ug/m3")
        return [field], ([0.0, 0.0], [0.0, 1.0]) if with_grid else None

    monkeypatch.setattr(hrrr, "_fetch_fields", fetch)
    monkeypatch.setattr(hrrr, "decode_messages", decode)
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})
    kwargs = {"max_fhr": 2, "smoke_every": 1, "met_every": 1, "met_vars": [], "workers": 1}
    day = date(2023, 8, 1)
    first = hrrr.extract_cycle(day, points, **kwargs)
    calls.clear()
    pd.testing.assert_frame_equal(hrrr.extract_cycle(day, points, **kwargs), first)
    assert not calls
    if change == "points":
        points = pd.DataFrame({"point_id": ["B"], "lat": [0.0], "lon": [1.0]})
    elif change == "coordinates":
        points.loc[0, "lon"] = 1.0
    elif change == "legacy":
        first.attrs.clear()
        first.to_parquet(tmp_path / "20230801_12z.parquet", index=False)
    elif change == "fields":
        kwargs["met_vars"] = [hrrr.DEFAULT_MET[0]]
    elif change == "horizon":
        kwargs["max_fhr"] = 3
    elif change == "smoke_interval":
        kwargs["smoke_every"] = 2
    else:
        kwargs["met_every"] = 2
    result = hrrr.extract_cycle(day, points, **kwargs)
    assert calls, "a different request must not reuse the cached extraction"
    if change == "points":
        assert set(result["point_id"]) == {"B"}
        assert set(result["value"]) == {20.0}
    elif change == "coordinates":
        assert set(result["point_id"]) == {"A"}
        assert set(result["value"]) == {20.0}
    elif change == "fields":
        assert set(result["var"]) == {"smoke", "u10"}
    elif change == "horizon":
        assert result["fhr"].max() == 3
    elif change == "smoke_interval":
        assert set(result["fhr"]) == {0, 2}
    loaded = hrrr.load_cycles([day])
    pd.testing.assert_frame_equal(loaded.drop(columns="issue_date"), result)
