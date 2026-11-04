from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.data import grib, hrrr
from serein.data.grib import GribField


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
