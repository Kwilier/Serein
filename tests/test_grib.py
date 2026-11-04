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
