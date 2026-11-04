from datetime import date
from unittest.mock import Mock

import pandas as pd
import pytest

from serein.data import hms


@pytest.mark.parametrize("fire", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_download_closes_response(tmp_path, monkeypatch, fire, failure):
    response = Mock()
    response.content = b"smoke archive"
    response.text = (
        "YearDay,Time,Lat,Lon,FRP,Satellite,Method\n"
        "2023213,1230,38.5,-121.5,12.0,VIIRS,manual\n"
    )
    monkeypatch.setattr(hms, "raw_dir", lambda name: tmp_path)
    monkeypatch.setattr(hms, "get", lambda *args, **kwargs: response)
    error = OSError("disk unavailable")
    if failure:
        def fail_write(*args, **kwargs):
            raise error

        if fire:
            monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)
        else:
            monkeypatch.setattr(type(tmp_path), "write_bytes", fail_write)
    download = hms.download_fire_day if fire else hms.download_day
    if failure:
        with pytest.raises(OSError) as caught:
            download(date(2023, 8, 1))
        assert caught.value is error
    else:
        path = download(date(2023, 8, 1))
        if fire:
            row = pd.read_parquet(path).iloc[0]
            assert row["latitude"] == 38.5
            assert row["frp"] == 12.0
            assert row["time_utc"] == pd.Timestamp("2023-08-01T12:30:00")
        else:
            assert path.read_bytes() == b"smoke archive"
    response.close.assert_called_once_with()


def test_fire_download_closes_response_before_parse_failure(tmp_path, monkeypatch):
    response = Mock(text="invalid archive")
    monkeypatch.setattr(hms, "raw_dir", lambda name: tmp_path)
    monkeypatch.setattr(hms, "get", lambda *args, **kwargs: response)
    with pytest.raises(KeyError):
        hms.download_fire_day(date(2023, 8, 1))
    response.close.assert_called_once_with()
