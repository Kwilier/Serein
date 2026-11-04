from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.data import aqm, hrrr


@pytest.mark.parametrize("source", [aqm, hrrr])
@pytest.mark.parametrize("operation", ["extract", "load"])
@pytest.mark.parametrize("cycle, error", [
    (-1, ValueError), (24, ValueError), (True, TypeError),
    (12.5, TypeError), (12.0, TypeError), ("12", TypeError),
    (np.nan, TypeError), (np.inf, TypeError),
])
def test_invalid_cycle_rejected_before_io(monkeypatch, source, operation, cycle, error):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid cycle must not access cache or network")

    monkeypatch.setattr(source, "interim_dir", unexpected)
    monkeypatch.setattr(source, "get", unexpected)
    points = pd.DataFrame({"point_id": ["A"], "lat": [40.0], "lon": [-120.0]})
    with pytest.raises(error, match="cycle"):
        if operation == "extract":
            source.extract_cycle(date(2023, 8, 1), points, cycle=cycle)
        else:
            source.load_cycles([date(2023, 8, 1)], cycle=cycle)


@pytest.mark.parametrize("source", [aqm, hrrr])
@pytest.mark.parametrize("cycle", [0, 23, np.int64(12)])
def test_valid_cycle_loads_matching_cache(tmp_path, monkeypatch, source, cycle):
    monkeypatch.setattr(source, "interim_dir", lambda name: tmp_path)
    expected = pd.DataFrame({"point_id": ["A"], "value": [8.0]})
    expected.to_parquet(tmp_path / f"20230801_{cycle:02d}z.parquet", index=False)
    actual = source.load_cycles([date(2023, 8, 1)], cycle=cycle)
    assert actual["value"].tolist() == [8.0]
