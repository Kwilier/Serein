from datetime import date

import numpy as np
import pandas as pd

from serein.data import grib, hrrr
from serein.data.grib import GribField


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
