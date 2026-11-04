from datetime import date
from types import SimpleNamespace

from serein.config import Region
from serein.data import firms


def test_download_cache_separates_regions(monkeypatch, tmp_path):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        lat = 38 if len(calls) == 1 else 45
        return SimpleNamespace(text=(
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
