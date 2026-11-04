import warnings
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from serein.data import hms
from serein.features.prepare import Prep


def test_missing_smoke_map_is_unknown_while_empty_map_is_clear(monkeypatch):
    points = pd.DataFrame({"lat": [38.], "lon": [-122.]})
    monkeypatch.setattr(hms, "read_day", lambda day: None)
    assert np.isnan(hms.density_at_points(date(2023, 8, 1), points)).all()

    monkeypatch.setattr(hms, "read_day", lambda day: gpd.GeoDataFrame())
    np.testing.assert_array_equal(hms.density_at_points(date(2023, 8, 1), points), [0.])


def test_available_smoke_map_uses_maximum_covering_density(monkeypatch):
    plumes = gpd.GeoDataFrame({"density_level": [1, 3]},
                             geometry=[box(-123, 37, -121, 39), box(-122.5, 37.5, -121.5, 38.5)],
                             crs=4326)
    monkeypatch.setattr(hms, "read_day", lambda day: plumes)
    points = pd.DataFrame({"lat": [38., 38., 40.], "lon": [-122., -122.75, -125.]})
    np.testing.assert_array_equal(hms.density_at_points(date(2023, 8, 1), points), [3., 1., 0.])


def test_unknown_smoke_days_do_not_define_clean_air_background():
    train = pd.DataFrame({"point_id": ["A", "A", "B"], "hms_prev": [np.nan, 0., np.nan],
                          "y_L0": [100., 5., 80.]})
    prep = Prep().fit(train)
    assert np.expm1(prep.site_bg["A"]) == 5.
    assert "B" not in prep.site_bg
    transformed, names = prep.transform(train)
    missing = names.index("hms_prev__missing")
    np.testing.assert_array_equal(transformed[:, missing], [1., 0., 1.])


def test_entirely_missing_smoke_history_has_explicit_missing_features():
    train = pd.DataFrame({"point_id": ["A", "A"], "hms_prev": [np.nan, np.nan],
                          "y_L0": [10., 20.]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        prep = Prep().fit(train)
        transformed, names = prep.transform(train)
    assert np.isfinite(transformed).all()
    assert not prep.site_bg
    for feature in ("hms_prev", "site_bg"):
        np.testing.assert_array_equal(transformed[:, names.index(feature + "__missing")], 1.)
