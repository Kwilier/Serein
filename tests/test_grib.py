import numpy as np
import pytest

from serein.data import grib


@pytest.mark.parametrize("blob", [
    b"GRIB",
    b"GRIB\0\0\0\2" + (40).to_bytes(8, "big") + b"7777",
    b"GRIB\0\0\0\2" + (16).to_bytes(8, "big"),
    b"GRIB\0\0\0\1" + (20).to_bytes(8, "big") + b"7777",
    b"GRIB\0\0\0\2" + (20).to_bytes(8, "big") + b"bad!",
])
def test_decode_rejects_invalid_message_before_native_decoder(monkeypatch, blob):
    def unexpected(message):
        pytest.fail("invalid GRIB must not reach the native decoder")

    monkeypatch.setattr(grib.eccodes, "codes_new_from_message", unexpected)
    with pytest.raises(ValueError, match="GRIB"):
        grib.decode_messages(blob)


def test_decode_reads_concatenated_messages_with_native_decoder():
    handle = grib.eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    try:
        count = grib.eccodes.codes_get_long(handle, "numberOfDataPoints")
        grib.eccodes.codes_set_values(handle, np.full(count, 12.0))
        first = grib.eccodes.codes_get_message(handle)
        grib.eccodes.codes_set_values(handle, np.full(count, 24.0))
        second = grib.eccodes.codes_get_message(handle)
    finally:
        grib.eccodes.codes_release(handle)

    fields, grid = grib.decode_messages(b"header\r\n" + first + second, with_grid=True)
    assert len(fields) == 2
    np.testing.assert_array_equal(fields[0].values, np.full(count, 12.0))
    np.testing.assert_array_equal(fields[1].values, np.full(count, 24.0))
    assert len(grid[0]) == len(grid[1]) == count
    with pytest.raises(ValueError, match="GRIB"):
        grib.decode_messages(first + second[:-1])


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


def test_point_index_radius_uses_surface_distance_including_cached_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    for _ in range(2):
        index = grib.PointIndex("test", [0.0], [0.0], [0.0], [90.0], max_km=10000.0)
        assert index.dist_km[0] == pytest.approx(np.pi * 6371.0 / 2)
        assert np.isnan(index.sample(np.array([10.0]))).all()


@pytest.mark.parametrize("radius", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_point_index_rejects_invalid_radius_before_cache_access(monkeypatch, radius):
    def unexpected(name):
        pytest.fail("invalid radius must not access the cache")

    monkeypatch.setattr(grib, "interim_dir", unexpected)
    with pytest.raises(ValueError, match="max_km"):
        grib.PointIndex("test", [0.0], [0.0], [0.0], [0.0], max_km=radius)


def test_point_index_zero_radius_accepts_only_exact_grid_point(tmp_path, monkeypatch):
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    index = grib.PointIndex("test", [0.0], [0.0], [0.0, 0.0], [0.0, 0.01], max_km=0.0)
    np.testing.assert_array_equal(index.sample(np.array([12.0])), [12.0, np.nan])


@pytest.mark.parametrize("lat,lon", [
    ([91.0], [0.0]), ([-91.0], [0.0]), ([0.0], [181.0]),
    ([0.0], [-181.0]), ([np.nan], [0.0]), ([0.0], [np.inf]),
    ([], []), ([[0.0]], [[0.0]]), ([0.0, 1.0], [0.0]),
])
@pytest.mark.parametrize("target", ["grid", "points"])
def test_point_index_rejects_invalid_coordinates_before_cache_access(
        monkeypatch, lat, lon, target):
    def unexpected(name):
        pytest.fail("invalid coordinates must not access the cache")

    monkeypatch.setattr(grib, "interim_dir", unexpected)
    grid = (lat, lon) if target == "grid" else ([0.0], [0.0])
    points = (lat, lon) if target == "points" else ([0.0], [0.0])
    with pytest.raises(ValueError, match="Point coordinates"):
        grib.PointIndex("test", *grid, *points)


def test_point_index_accepts_geographic_boundaries_and_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    lat, lon = [90.0, -90.0, 0.0, 0.0], [0.0, 0.0, -180.0, 180.0]
    for _ in range(2):
        index = grib.PointIndex("test", lat, lon, lat, lon)
        np.testing.assert_allclose(index.dist_km, 0.0, atol=1e-9)
        np.testing.assert_array_equal(index.sample(np.ones(4)), np.ones(4))


@pytest.mark.parametrize("idx,dist", [
    ([-1], [0.0]), ([2], [0.0]), ([0.5], [0.0]), ([True], [0.0]),
    ([[0]], [0.0]), ([0, 1], [0.0]), ([0], [[0.0]]),
    ([0], [-1.0]), ([0], [np.nan]), ([0], [np.inf]),
    ([0], [2 * 6371.0 + 1]),
])
def test_point_index_rejects_invalid_cached_lookup(tmp_path, monkeypatch, idx, dist):
    monkeypatch.setattr(grib, "interim_dir", lambda name: tmp_path)
    grib.PointIndex("test", [0.0, 0.0], [0.0, 1.0], [0.0], [0.0])
    cache, = tmp_path.glob("*.npz")
    np.savez(cache, idx=idx, dist_km=dist)

    with pytest.raises(ValueError, match="Invalid grid index cache"):
        grib.PointIndex("test", [0.0, 0.0], [0.0, 1.0], [0.0], [0.0])
