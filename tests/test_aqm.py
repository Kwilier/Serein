from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
