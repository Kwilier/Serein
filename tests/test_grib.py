import numpy as np

from serein.data import grib


def test_point_index_cache_distinguishes_grid_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    first = grib.PointIndex("test", [0.0, 0.0], [0.0, 1.0], [0.0], [0.0])
    moved = grib.PointIndex("test", [0.0, 0.0], [1.0, 0.0], [0.0], [0.0])

    np.testing.assert_array_equal(first.sample(np.array([10.0, 20.0])), [10.0])
    np.testing.assert_array_equal(moved.sample(np.array([20.0, 10.0])), [10.0])


def test_point_index_cached_distances_use_current_radius(tmp_path, monkeypatch):
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    wide = grib.PointIndex("test", [0.0], [0.0], [0.0], [0.05], max_km=10.0)
    narrow = grib.PointIndex("test", [0.0], [0.0], [0.0], [0.05], max_km=1.0)

    np.testing.assert_array_equal(wide.sample(np.array([10.0])), [10.0])
    assert np.isnan(narrow.sample(np.array([10.0]))).all()
