from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.data import aqm, grib, hrrr


@pytest.mark.parametrize("module", [aqm, hrrr])
@pytest.mark.parametrize("lat,lon", [
    ([], []), ([np.nan], [0.0]), ([0.0], [np.inf]),
    ([91.0], [0.0]), ([-91.0], [0.0]), ([0.0], [181.0]), ([0.0], [-181.0]),
])
def test_invalid_points_fail_before_io(monkeypatch, module, lat, lon):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid points must fail before cache or network access")

    monkeypatch.setattr(module, "interim_dir", unexpected)
    points = pd.DataFrame({"point_id": list(range(len(lat))), "lat": lat, "lon": lon})
    with pytest.raises(ValueError, match="coordinates"):
        module.extract_cycle(date(2023, 8, 1), points)


def test_geographic_boundaries_are_valid():
    grib.validate_point_coordinates([-90.0, 0.0, 90.0], [-180.0, 0.0, 180.0])
