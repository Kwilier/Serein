from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from serein.data import aqm, hrrr
from serein.data.grib import GribField


@pytest.mark.parametrize("fields", [[], [GribField(
    "pm25", "surface", 0, 24, np.array([10.0]), "ug/m3")]])
def test_aqm_skips_cycles_without_hourly_fields(tmp_path, monkeypatch, fields):
    monkeypatch.setattr(aqm, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(aqm, "get", lambda *args, **kwargs: SimpleNamespace(content=b"cycle"))
    monkeypatch.setattr(aqm, "decode_messages", lambda *args, **kwargs: (fields, None))
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})

    assert aqm.extract_cycle(date(2023, 8, 1), points) is None
    assert not list(tmp_path.glob("*.parquet"))


@pytest.mark.parametrize("blobs", [[], [(hrrr.SMOKE, b"empty")]])
def test_hrrr_skips_cycles_without_decoded_fields(tmp_path, monkeypatch, blobs):
    monkeypatch.setattr(hrrr, "interim_dir", lambda name: tmp_path)
    monkeypatch.setattr(hrrr, "_fetch_fields", lambda *args: blobs)
    monkeypatch.setattr(hrrr, "decode_messages", lambda *args, **kwargs: ([], None))
    points = pd.DataFrame({"point_id": ["A"], "lat": [0.0], "lon": [0.0]})

    assert hrrr.extract_cycle(date(2023, 8, 1), points, max_fhr=0, workers=1) is None
    assert not list(tmp_path.glob("*.parquet"))
