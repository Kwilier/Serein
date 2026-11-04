import numpy as np
import pandas as pd
import pytest

from serein.config import Region
from serein.data.points import build_points, grid_points

REGION = Region(lat_min=38.0, lat_max=39.0, lon_min=-122.0, lon_max=-121.0)


def sites():
    return pd.DataFrame({"site_id": ["EPA1"], "lat": [38.0], "lon": [-122.0],
                         "gmt_offset": [-8]})


@pytest.mark.parametrize("step", [0.0, -0.5, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("builder", [grid_points, build_points])
def test_invalid_grid_spacing_is_rejected(step, builder):
    with pytest.raises(ValueError, match="step must be finite and positive"):
        if builder is grid_points:
            builder(REGION, step)
        else:
            builder(sites(), REGION, step)


def test_grid_contains_region_endpoints_and_preserves_sensors():
    points = build_points(sites(), REGION, 0.5)
    grid = points[points["kind"] == "grid"]
    assert set(zip(grid["lat"], grid["lon"])) == {
        (lat, lon) for lat in (38.0, 38.5, 39.0) for lon in (-122.0, -121.5, -121.0)
    }
    assert points["point_id"].is_unique
    assert (points["gmt_offset"] == -8).all()
    assert list(points[points["kind"] == "epa"]["point_id"]) == ["EPA1"]


def test_none_spacing_disables_grid():
    points = build_points(sites(), REGION, None)
    assert list(points["point_id"]) == ["EPA1"]


@pytest.mark.parametrize("step", [0.0005, 0.00005])
def test_fine_grid_keeps_every_location(step):
    region = Region(lat_min=38.0, lat_max=38.001,
                    lon_min=-122.0, lon_max=-121.999)
    grid = grid_points(region, step)
    points = build_points(sites(), region, step)
    assert grid["point_id"].is_unique
    assert not grid.duplicated(["lat", "lon"]).any()
    assert len(points) == len(grid) + 1
    np.testing.assert_allclose(np.diff(np.sort(grid["lat"].unique())), step)
    np.testing.assert_allclose(np.diff(np.sort(grid["lon"].unique())), step)


def test_standard_grid_ids_remain_stable():
    grid = grid_points(REGION, 0.5)
    assert grid.iloc[0]["point_id"] == "G+038.000-0122.000"


@pytest.mark.parametrize("field,value", [("lat", 38.5), ("lon", -121.5),
                                          ("gmt_offset", -7),
                                          ("reporting_area", "Other area")])
def test_conflicting_epa_ids_are_rejected(field, value):
    original = sites().assign(reporting_area="Area")
    duplicate = original.copy()
    duplicate[field] = value
    with pytest.raises(ValueError, match="conflicting.*point_id"):
        build_points(pd.concat([original, duplicate]), REGION, None)


def test_conflicting_purpleair_ids_are_rejected():
    sensors = pd.DataFrame({"sensor_index": [123, 123], "lat": [38.0, 38.5],
                            "lon": [-122.0, -122.0]})
    with pytest.raises(ValueError, match="conflicting.*point_id"):
        build_points(sites(), REGION, None, sensors)


def test_identical_sensor_rows_are_deduplicated():
    original = sites()
    sensors = pd.DataFrame({"sensor_index": [123, 123], "lat": [38.5, 38.5],
                            "lon": [-121.5, -121.5]})
    points = build_points(pd.concat([original, original]), REGION, None, sensors)
    assert list(points["point_id"]) == ["EPA1", "PA123"]


def test_ids_cannot_collide_between_sensor_kinds():
    original = sites().assign(site_id="PA123")
    sensors = pd.DataFrame({"sensor_index": [123], "lat": [38.0], "lon": [-122.0]})
    with pytest.raises(ValueError, match="conflicting.*point_id"):
        build_points(original, REGION, None, sensors)
