from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from serein.config import Region
from serein.data import firms


def test_download_cache_separates_regions(monkeypatch, tmp_path):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        lat = 38 if len(calls) == 1 else 45
        return SimpleNamespace(close=Mock(), text=(
            "latitude,longitude,acq_date,acq_time,frp\n"
            f"{lat},-122,2023-07-01,1200,10\n"
        ))

    monkeypatch.setattr(firms, "raw_dir", lambda source: tmp_path)
    monkeypatch.setattr(firms, "_key", lambda: "test-placeholder")
    monkeypatch.setattr(firms, "get", get)
    monkeypatch.setattr(firms, "SOURCES_SP", ["VIIRS_SNPP_SP"])
    day = date(2023, 7, 1)
    south = Region(lat_min=37, lat_max=39, lon_min=-123, lon_max=-121)
    north = Region(lat_min=44, lat_max=46, lon_min=-123, lon_max=-121)
    first = firms.download_range(day, day, south)
    second = firms.download_range(day, day, north)
    cached = firms.download_range(day, day, north)

    assert len(calls) == 2
    assert first.latitude.tolist() == [38]
    assert second.latitude.tolist() == [45]
    assert cached.equals(second)
    assert len(firms.load_cached()) == 2


@pytest.mark.parametrize("body", [
    "latitude,longitude,acq_date,acq_time,frp\n38,-122,2023-07-01,1200,10\n",
    "", "  \n", "Invalid request",
])
def test_download_closes_response(monkeypatch, tmp_path, body):
    response = Mock(text=body)
    monkeypatch.setattr(firms, "raw_dir", lambda source: tmp_path)
    monkeypatch.setattr(firms, "_key", lambda: "test-placeholder")
    monkeypatch.setattr(firms, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(firms, "SOURCES_SP", ["VIIRS_SNPP_SP"])
    day = date(2023, 7, 1)
    region = Region(lat_min=37, lat_max=39, lon_min=-123, lon_max=-121)
    result = firms.download_range(day, day, region)
    assert result.empty == (not body.strip() or body.startswith("Invalid"))
    if not result.empty:
        assert result.frp.tolist() == [10]
    response.close.assert_called_once_with()


def test_download_closes_response_on_text_failure(monkeypatch, tmp_path):
    from unittest.mock import PropertyMock

    response = Mock()
    error = OSError("response read failed")
    type(response).text = PropertyMock(side_effect=error)
    monkeypatch.setattr(firms, "raw_dir", lambda source: tmp_path)
    monkeypatch.setattr(firms, "_key", lambda: "test-placeholder")
    monkeypatch.setattr(firms, "get", lambda *args, **kwargs: response)
    day = date(2023, 7, 1)
    region = Region(lat_min=37, lat_max=39, lon_min=-123, lon_max=-121)
    with pytest.raises(OSError) as caught:
        firms.download_range(day, day, region)
    assert caught.value is error
    response.close.assert_called_once_with()
