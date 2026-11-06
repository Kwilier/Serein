import hashlib
import json
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import MultiPolygon, Point, Polygon, box

from serein.spatial import (
    AREA_CRS,
    aggregate_tracts,
    degree_grid_cells,
    forecast_neighborhood,
    grid_registry,
    main,
    metro_grid,
    plot_cell_forecasts,
    tract_weights,
)


@pytest.fixture
def cells():
    return gpd.GeoDataFrame({"point_id": ["a", "b", "c"]},
                            geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
                                      box(2000, 0, 3000, 1000)], crs=AREA_CRS)


@pytest.fixture
def tracts():
    return gpd.GeoDataFrame({"GEOID": ["001", "002", "003"]},
                            geometry=[box(500, 0, 2000, 1000), box(2000, 0, 4000, 1000),
                                      box(5000, 0, 6000, 1000)], crs=AREA_CRS)


@pytest.fixture
def forecasts():
    return pd.DataFrame({"point_id": ["a", "b", "c"], "kind": "grid",
                         "issue_date": date(2023, 8, 18), "lead": 1,
                         "valid_date": date(2023, 8, 19), "q50": [10., 40., 70.]})


def test_intersection_weights_use_partial_cell_areas(cells, tracts):
    weights = tract_weights(cells, tracts).set_index(["tract_id", "point_id"])
    assert set(weights.index) == {("001", "a"), ("001", "b"), ("002", "c")}
    assert weights.loc[("001", "a"), "area_m2"] == pytest.approx(500_000)
    assert weights.loc[("001", "b"), "area_m2"] == pytest.approx(1_000_000)


def test_area_weighted_forecast_keeps_partial_and_absent_tracts(cells, tracts, forecasts):
    result = aggregate_tracts(forecasts, cells, tracts).set_index("tract_id")
    assert result.loc["001", "q50_area_mean"] == pytest.approx(30.)
    assert result.loc["001", "forecast_coverage"] == pytest.approx(1.)
    assert result.loc["001", "n_available"] == 2
    assert result.loc["001", "status"] == "ok"
    assert result.loc["002", "grid_coverage"] == pytest.approx(.5)
    assert result.loc["002", "forecast_coverage"] == pytest.approx(.5)
    assert result.loc["002", "status"] == "insufficient_coverage"
    assert np.isnan(result.loc["002", "q50_area_mean"])
    assert result.loc["003", "grid_coverage"] == 0
    assert result.loc["003", "status"] == "no_forecast"
    assert result.loc["003", "n_cells"] == 0
    assert not {"q05", "q95", "aqi_cat"}.intersection(result.columns)
    assert result.crs == tracts.crs


def test_coverage_threshold_is_explicit_and_inclusive(cells, tracts, forecasts):
    result = aggregate_tracts(forecasts, cells, tracts, min_coverage=.5).set_index("tract_id")
    assert result.loc["002", "q50_area_mean"] == 70
    assert result.loc["002", "min_coverage"] == .5
    assert result.loc["003", "status"] == "no_forecast"


@pytest.mark.parametrize("missing", ["row", "nan", "nullable"])
def test_missing_forecasts_reduce_available_area_without_filling(cells, tracts, forecasts, missing):
    if missing == "row":
        forecasts = forecasts[forecasts.point_id != "b"]
    elif missing == "nullable":
        forecasts["q50"] = forecasts.q50.astype("Float64")
        forecasts.loc[1, "q50"] = pd.NA
    else:
        forecasts.loc[1, "q50"] = np.nan
    result = aggregate_tracts(forecasts, cells, tracts).set_index("tract_id")
    assert result.loc["001", "grid_coverage"] == 1
    assert result.loc["001", "forecast_coverage"] == pytest.approx(1 / 3)
    assert result.loc["001", "n_cells"] == 2
    assert result.loc["001", "n_available"] == 1
    assert np.isnan(result.loc["001", "q50_area_mean"])


def test_all_missing_forecasts_still_produce_auditable_rows(cells, tracts, forecasts):
    forecasts["q50"] = np.nan
    result = aggregate_tracts(forecasts, cells, tracts)
    assert len(result) == len(tracts)
    assert result.q50_area_mean.isna().all()
    assert result.status.eq("no_forecast").all()
    assert result.forecast_coverage.eq(0).all()


def test_each_issue_and_lead_has_its_own_population(cells, tracts, forecasts):
    later = forecasts.iloc[[0]].copy()
    later["lead"] = 2
    later["valid_date"] = date(2023, 8, 20)
    later["q50"] = 100.
    result = aggregate_tracts(pd.concat([later, forecasts]), cells, tracts)
    assert len(result) == 6
    first = result[(result.tract_id == "001") & (result.lead == 1)].iloc[0]
    second = result[(result.tract_id == "001") & (result.lead == 2)].iloc[0]
    assert first.q50_area_mean == pytest.approx(30)
    assert np.isnan(second.q50_area_mean)
    assert second.n_available == 1


def test_sensor_forecasts_are_excluded(cells, tracts, forecasts):
    sensor = forecasts.iloc[[0]].copy()
    sensor["point_id"], sensor["kind"], sensor["q50"] = "epa", "epa", -999.
    expected = aggregate_tracts(forecasts, cells, tracts)
    actual = aggregate_tracts(pd.concat([forecasts, sensor]), cells, tracts)
    pd.testing.assert_frame_equal(actual, expected)


def test_reprojected_inputs_keep_areas_and_output_crs(cells, tracts, forecasts):
    expected = aggregate_tracts(forecasts, cells, tracts)
    actual = aggregate_tracts(forecasts, cells.to_crs(4326), tracts.to_crs(4326))
    assert actual.crs.to_epsg() == 4326
    np.testing.assert_allclose(actual.tract_area_m2, expected.tract_area_m2, rtol=1e-8)
    np.testing.assert_allclose(actual.q50_area_mean, expected.q50_area_mean, equal_nan=True)


def test_holes_and_multipart_tracts_use_only_polygon_area(cells, forecasts):
    outer = box(0, 0, 2000, 1000)
    hole = box(100, 100, 900, 900)
    shape = Polygon(outer.exterior.coords, holes=[hole.exterior.coords])
    multipart = MultiPolygon([box(0, 0, 1000, 1000), box(2000, 0, 3000, 1000)])
    tracts = gpd.GeoDataFrame({"GEOID": ["hole", "multipart"]},
                              geometry=[shape, multipart], crs=AREA_CRS)
    result = aggregate_tracts(forecasts, cells, tracts).set_index("tract_id")
    expected_hole_mean = (10 * (outer.area / 2 - hole.area) + 40 * outer.area / 2) / shape.area
    assert result.loc["hole", "q50_area_mean"] == pytest.approx(expected_hole_mean)
    assert result.loc["multipart", "q50_area_mean"] == pytest.approx(40)
    assert result.forecast_coverage.eq(1).all()


def test_constant_field_and_subdivision_conserve_the_mean(cells, forecasts):
    forecasts["q50"] = 42.
    tracts = gpd.GeoDataFrame({"GEOID": ["whole", "left", "right"]},
                              geometry=[box(0, 0, 3000, 1000), box(0, 0, 1250, 1000),
                                        box(1250, 0, 3000, 1000)], crs=AREA_CRS)
    result = aggregate_tracts(forecasts, cells, tracts)
    np.testing.assert_allclose(result.q50_area_mean, 42)
    np.testing.assert_allclose(result.forecast_coverage, 1)


def test_very_large_finite_concentrations_do_not_overflow(cells, tracts, forecasts):
    forecasts["q50"] = np.finfo(float).max / 2
    result = aggregate_tracts(forecasts, cells, tracts)
    assert np.isfinite(result.iloc[0].q50_area_mean)
    assert result.iloc[0].q50_area_mean == pytest.approx(np.finfo(float).max / 2)


@pytest.mark.parametrize("column,value,match", [
    ("point_id", "unknown", "absent"), ("point_id", None, "identifiers"),
    ("point_id", "", "identifiers"), ("q50", -1., "q50"),
    ("q50", np.inf, "q50"), ("q50", -np.inf, "q50"),
    ("lead", -1, "lead"), ("lead", 1.5, "lead"), ("lead", np.nan, "lead"),
    ("issue_date", None, "calendar dates"), ("valid_date", None, "calendar dates"),
    ("issue_date", "2023-08-18 01:00:00", "calendar dates"),
    ("valid_date", "2023-08-19T00:00:00Z", "calendar dates"),
    ("valid_date", date(2023, 8, 20), "plus lead"),
])
def test_invalid_forecasts_are_rejected(cells, tracts, forecasts, column, value, match):
    forecasts[column] = forecasts[column].astype(object)
    forecasts.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        aggregate_tracts(forecasts, cells, tracts)


def test_duplicate_forecasts_are_rejected(cells, tracts, forecasts):
    with pytest.raises(ValueError, match="duplicate grid forecasts"):
        aggregate_tracts(pd.concat([forecasts, forecasts.iloc[[0]]]), cells, tracts)


@pytest.mark.parametrize("threshold", [0, -1, 1.1, np.nan, np.inf, True])
def test_invalid_coverage_thresholds_are_rejected(cells, tracts, forecasts, threshold):
    with pytest.raises(ValueError, match="min_coverage"):
        aggregate_tracts(forecasts, cells, tracts, min_coverage=threshold)


@pytest.mark.parametrize("table", ["cells", "tracts"])
@pytest.mark.parametrize("problem", ["missing_crs", "duplicate", "null_id", "numeric_id",
                                      "empty", "point", "null_geometry", "invalid"])
def test_invalid_geometry_tables_are_rejected(cells, tracts, forecasts, table, problem):
    frame = cells.copy() if table == "cells" else tracts.copy()
    identifier = "point_id" if table == "cells" else "GEOID"
    if problem == "missing_crs":
        frame = frame.set_crs(None, allow_override=True)
    elif problem == "duplicate":
        frame.loc[1, identifier] = frame.loc[0, identifier]
    elif problem == "null_id":
        frame.loc[0, identifier] = None
    elif problem == "numeric_id":
        frame[identifier] = frame[identifier].astype(object)
        frame.loc[0, identifier] = 1
    elif problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "point":
        frame.loc[0, "geometry"] = Point(0, 0)
    elif problem == "null_geometry":
        frame.loc[0, "geometry"] = None
    else:
        frame.loc[0, "geometry"] = Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])
    with pytest.raises(ValueError):
        aggregate_tracts(forecasts, frame if table == "cells" else cells,
                         frame if table == "tracts" else tracts)


def test_overlapping_cells_are_rejected_but_shared_edges_are_allowed(cells, tracts):
    assert len(tract_weights(cells, tracts)) == 3
    cells.loc[1, "geometry"] = box(500, 0, 1500, 1000)
    with pytest.raises(ValueError, match="overlap"):
        tract_weights(cells, tracts)


def test_degree_grid_preserves_gaps_and_ignores_sensors():
    points = pd.DataFrame({"point_id": ["a", "b", "epa"], "kind": ["grid", "grid", "epa"],
                           "lat": [38., 38., 38.], "lon": [-122., -121.5, -121.75]})
    cells = degree_grid_cells(points, .25, .25)
    assert list(cells.point_id) == ["a", "b"]
    assert cells.geometry.iloc[0].bounds == (-122.125, 37.875, -121.875, 38.125)
    assert cells.geometry.iloc[0].distance(cells.geometry.iloc[1]) == .25


@pytest.mark.parametrize("bounds", [[], [-122, 38, -122, 39], [-122, 39, -121, 38],
                                     [-126, 38, -121, 39], [-122, 23, -121, 39],
                                     [-122, 38, np.inf, 39]])
def test_invalid_metro_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match="bounds"):
        metro_grid(bounds)


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, True, .0000001, 1.0000001])
def test_invalid_metro_spacing_is_rejected(spacing):
    with pytest.raises(ValueError, match="spacing_km"):
        metro_grid([-122, 38, -121.9, 38.1], spacing)


def test_metro_grid_has_stable_ids_meter_cells_and_centers():
    small = metro_grid([-121.55, 38.5, -121.5, 38.55], 1)
    big = metro_grid([-121.6, 38.45, -121.45, 38.6], 1)
    assert small.crs.to_epsg() == 5070
    assert small.point_id.is_unique
    assert small.point_id.isin(big.point_id).all()
    pd.testing.assert_frame_equal(small.set_index("point_id"),
                                   big.set_index("point_id").loc[small.point_id])
    np.testing.assert_allclose(small.area, 1_000_000)
    centers = gpd.GeoSeries(gpd.points_from_xy(small.lon, small.lat), crs=4326).to_crs(5070)
    np.testing.assert_allclose(centers.x, small.geometry.centroid.x, atol=1e-7)
    np.testing.assert_allclose(centers.y, small.geometry.centroid.y, atol=1e-7)
    assert small.kind.eq("grid").all()
    assert small.gmt_offset.eq(-8).all()


def test_metro_grid_uses_nearest_monitor_time_zone():
    sites = pd.DataFrame({"lat": [38.5], "lon": [-121.5], "gmt_offset": [-7]})
    cells = metro_grid([-121.55, 38.5, -121.5, 38.55], sites=sites)
    assert cells.gmt_offset.eq(-7).all()


def test_metro_grid_rejects_excessive_allocation_before_building_cells():
    with pytest.raises(ValueError, match="exceeding max_cells"):
        metro_grid([-125, 24, -66, 50], 1)


def test_metro_grid_to_tracts_roundtrip():
    cells = metro_grid([-121.55, 38.5, -121.5, 38.55], 2)
    tracts = gpd.GeoDataFrame({"GEOID": ["synthetic"]}, geometry=[cells.geometry.union_all()], crs=cells.crs)
    forecasts = pd.DataFrame({"point_id": cells.point_id, "q50": np.arange(len(cells)) + 1.,
                              "issue_date": "2023-08-18", "lead": 1, "valid_date": "2023-08-19"})
    result = aggregate_tracts(forecasts, cells, tracts).iloc[0]
    assert result.q50_area_mean == pytest.approx(forecasts.q50.mean())
    assert result.forecast_coverage == pytest.approx(1)
    assert result.n_available == len(cells)


def test_cli_grid_writes_readable_geoparquet(tmp_path):
    output = tmp_path / "grid.parquet"
    assert main(["grid", "--bounds", "-121.55", "38.5", "-121.5", "38.55",
                 "--spacing-km", "2", "--output", str(output)]) == 0
    cells = gpd.read_parquet(output)
    assert cells.crs.to_epsg() == 5070
    assert not cells.empty
    assert cells.spacing_km.eq(2).all()


def test_cli_tracts_reads_local_polygons_and_preserves_geoid(tmp_path, cells, tracts, forecasts):
    cell_path, tract_path, forecast_path, output = [tmp_path / name for name in
        ["cells.parquet", "tracts.geojson", "forecasts.parquet", "summary.parquet"]]
    cells.to_parquet(cell_path)
    tracts.to_crs(4326).to_file(tract_path, driver="GeoJSON")
    forecasts.to_parquet(forecast_path)
    assert main(["tracts", "--cells", str(cell_path), "--tracts", str(tract_path),
                 "--forecasts", str(forecast_path), "--output", str(output)]) == 0
    result = gpd.read_parquet(output).set_index("tract_id")
    assert list(result.index) == ["001", "002", "003"]
    assert result.loc["001", "q50_area_mean"] == pytest.approx(30.)
    assert result.loc["003", "status"] == "no_forecast"


def test_cli_refuses_existing_output_before_reading_inputs(tmp_path):
    output = tmp_path / "keep.parquet"
    output.write_bytes(b"original")
    with pytest.raises(SystemExit) as error:
        main(["tracts", "--cells", "absent.parquet", "--tracts", "absent.geojson",
              "--forecasts", "absent.parquet", "--output", str(output)])
    assert error.value.code == 2
    assert output.read_bytes() == b"original"


def test_grid_registry_preserves_sensor_metadata_and_replaces_old_grid():
    points = pd.DataFrame({"point_id": ["epa", "pa", "old_grid"],
                           "kind": ["epa", "purpleair", "grid"], "lat": [38.5, 38.6, 38.7],
                           "lon": [-121.5, -121.6, -121.7], "gmt_offset": [-8, -8, -8],
                           "reporting_area": ["Sacramento", None, None],
                           "state": ["CA", None, None]})
    original = points.copy(deep=True)
    cells = metro_grid([-121.55, 38.5, -121.5, 38.55], sites=points)
    registry = grid_registry(cells, points)
    pd.testing.assert_frame_equal(registry.iloc[:2][points.columns].reset_index(drop=True),
                                   points.iloc[:2].reset_index(drop=True))
    pd.testing.assert_frame_equal(points, original)
    assert "old_grid" not in registry.point_id.to_list()
    assert registry.point_id.is_unique
    assert "geometry" not in registry
    assert set(registry.loc[registry.kind == "grid", "point_id"]) == set(cells.point_id)


def test_grid_registry_time_zone_comes_from_epa_not_old_grid():
    points = pd.DataFrame({"point_id": ["epa", "old_grid"], "kind": ["epa", "grid"],
                           "lat": [39., 38.51], "lon": [-121., -121.51], "gmt_offset": [-7, -8]})
    cells = metro_grid([-121.55, 38.5, -121.5, 38.55], sites=points)
    assert cells.gmt_offset.eq(-7).all()


def test_grid_registry_rejects_colliding_sensor_identifiers():
    cells = metro_grid([-121.55, 38.5, -121.5, 38.55])
    points = pd.DataFrame(cells.iloc[[0]].drop(columns="geometry"))
    points["kind"] = "epa"
    with pytest.raises(ValueError, match="duplicates"):
        grid_registry(cells, points)


def test_cli_grid_can_export_a_separate_registry(tmp_path):
    points = pd.DataFrame({"point_id": ["epa"], "kind": "epa", "lat": [38.5],
                           "lon": [-121.5], "gmt_offset": [-8], "state": "CA"})
    sites, output, registry = [tmp_path / name for name in
                              ["sites.parquet", "cells.parquet", "registry.parquet"]]
    points.to_parquet(sites)
    original = sites.read_bytes()
    assert main(["grid", "--bounds", "-121.55", "38.5", "-121.5", "38.55",
                 "--sites", str(sites), "--output", str(output),
                 "--registry-output", str(registry)]) == 0
    assert sites.read_bytes() == original
    exported = pd.read_parquet(registry)
    assert exported.point_id.iloc[0] == "epa"
    assert exported.kind.eq("grid").sum() == len(gpd.read_parquet(output))


@pytest.mark.parametrize("problem", ["existing", "same", "missing_parent", "no_sites"])
def test_cli_grid_checks_all_outputs_before_writing(tmp_path, problem):
    sites, output, registry = [tmp_path / name for name in
                              ["sites.parquet", "cells.parquet", "registry.parquet"]]
    pd.DataFrame({"point_id": ["epa"], "kind": "epa", "lat": [38.5],
                  "lon": [-121.5], "gmt_offset": [-8]}).to_parquet(sites)
    if problem == "existing":
        registry.write_bytes(b"keep")
    elif problem == "same":
        registry = output
    elif problem == "missing_parent":
        registry = tmp_path / "absent" / "points.parquet"
    args = ["grid", "--bounds", "-121.55", "38.5", "-121.5", "38.55", "--output", str(output),
            "--registry-output", str(registry)]
    if problem != "no_sites":
        args.extend(["--sites", str(sites)])
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert not output.exists()
    if problem == "existing":
        assert registry.read_bytes() == b"keep"


def test_mapped_forecasts_must_lie_inside_their_footprints(cells, tracts, forecasts):
    centers = cells.geometry.centroid.to_crs(4326)
    forecasts["lon"], forecasts["lat"] = centers.x, centers.y
    assert aggregate_tracts(forecasts, cells, tracts).iloc[0].status == "ok"
    forecasts.loc[0, "lon"] = forecasts.loc[0, "lon"] + 1.
    with pytest.raises(ValueError, match="outside their grid cell"):
        aggregate_tracts(forecasts, cells, tracts)


def test_random_rectangles_match_independent_overlap_arithmetic():
    rng = np.random.default_rng(42)
    ix, iy = np.meshgrid(np.arange(10), np.arange(12))
    x, y = ix.ravel() * 1000., iy.ravel() * 1000.
    cells = gpd.GeoDataFrame({"point_id": [f"g{i}" for i in range(len(x))]},
                             geometry=[box(a, b, a + 1000, b + 1000) for a, b in zip(x, y)], crs=AREA_CRS)
    n_tracts = 20
    lower = rng.uniform(-500, 10000, (n_tracts, 2))
    upper = lower + rng.uniform(100, 3000, (n_tracts, 2))
    tracts = gpd.GeoDataFrame({"GEOID": [f"t{i}" for i in range(n_tracts)]},
                              geometry=[box(*lo, *hi) for lo, hi in zip(lower, upper)], crs=AREA_CRS)
    values = rng.uniform(0, 150, len(cells))
    values[::4] = np.nan
    forecasts = pd.DataFrame({"point_id": cells.point_id, "q50": values,
                              "issue_date": "2023-08-18", "lead": 1, "valid_date": "2023-08-19"})
    result = aggregate_tracts(forecasts, cells, tracts, min_coverage=.01)
    for i, row in result.iterrows():
        width = np.maximum(0, np.minimum(x + 1000, upper[i, 0]) - np.maximum(x, lower[i, 0]))
        height = np.maximum(0, np.minimum(y + 1000, upper[i, 1]) - np.maximum(y, lower[i, 1]))
        areas = width * height
        area = np.prod(upper[i] - lower[i])
        available = np.isfinite(values) & (areas > 0)
        assert row.grid_coverage == pytest.approx(areas.sum() / area)
        assert row.forecast_coverage == pytest.approx(areas[available].sum() / area)
        if row.status == "ok":
            assert row.q50_area_mean == pytest.approx(np.average(values[available], weights=areas[available]))
        else:
            assert np.isnan(row.q50_area_mean)


@pytest.mark.parametrize("missing", ["none", "some", "all"])
def test_projected_map_draws_cell_polygons_including_missing_forecasts(tmp_path, cells, forecasts, monkeypatch, missing):
    import matplotlib.pyplot as plt

    if missing == "some":
        forecasts = forecasts.iloc[:1]
    elif missing == "all":
        forecasts["q50"] = np.nan
    subplots = plt.subplots
    captured = []

    def capture(*args, **kwargs):
        result = subplots(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(plt, "subplots", capture)
    output = tmp_path / "forecast.png"
    plot_cell_forecasts(forecasts, cells, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    fig, axes = captured[0]
    axis = axes.ravel()[0]
    assert len(axis.images) == 0
    assert sum(len(collection.get_paths()) for collection in axis.collections) == len(cells)
    assert "2023-08-18" in axis.get_title()
    assert not plt.fignum_exists(fig.number)
    assert (axis.get_legend() is not None) == (missing != "none")


def test_projected_map_rejects_excessive_panels_before_writing(tmp_path, cells, forecasts):
    later = forecasts.copy()
    later["lead"], later["valid_date"] = 2, date(2023, 8, 20)
    output = tmp_path / "forecast.png"
    with pytest.raises(ValueError, match="max_panels"):
        plot_cell_forecasts(pd.concat([forecasts, later]), cells, output, max_panels=1)
    assert not output.exists()


def test_projected_map_preserves_existing_output(tmp_path, cells, forecasts):
    output = tmp_path / "forecast.png"
    output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        plot_cell_forecasts(forecasts, cells, output)
    assert output.read_bytes() == b"original"


def test_cli_projected_map_writes_png(tmp_path, cells, forecasts):
    cell_path, forecast_path, output = [tmp_path / name for name in
                                        ["cells.parquet", "forecasts.parquet", "map.png"]]
    cells.to_parquet(cell_path)
    forecasts.to_parquet(forecast_path)
    assert main(["plot", "--cells", str(cell_path), "--forecasts", str(forecast_path),
                 "--output", str(output)]) == 0
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def prepared_neighborhood(tmp_path, cells, forecasts, monkeypatch):
    from serein.data import aqm, hrrr
    from serein.data import points as point_module
    from serein.models import infer

    config, cell_path, run, output = [tmp_path / name for name in
                                    ["study.yaml", "cells.parquet", "model", "export"]]
    config.write_text(yaml.safe_dump({"name": "synthetic", "split": {"calibrate": [2023]},
        "region": {"lon_min": -125, "lon_max": -66, "lat_min": 20, "lat_max": 50},
        "season_months": [8], "graph": {"k_neighbors": 2, "max_km": 300},
        "availability": {"hrrr_cycle": 12, "aqm_cycle": 12}}))
    cells.to_parquet(cell_path)
    run.mkdir()
    (run / "model.json").write_text(json.dumps({"node_ids": ["epa"], "n_members": 5, "leads": [1]}))
    centers = cells.geometry.centroid.to_crs(4326)
    registry = pd.DataFrame({"point_id": [*cells.point_id, "epa"], "kind": ["grid"] * len(cells) + ["epa"],
                              "lat": [*centers.y, centers.y.iloc[0]], "lon": [*centers.x, centers.x.iloc[0]]})
    registry["gmt_offset"] = 0.
    monkeypatch.setattr(point_module, "load_points", lambda name: registry.copy())
    point_hash = hashlib.sha256(pd.util.hash_pandas_object(
        registry[["point_id", "lat", "lon"]], index=False).to_numpy().tobytes()).hexdigest()
    cache = pd.DataFrame({"point_id": cells.point_id, "issue_date": date(2023, 8, 18),
                          "valid_utc": pd.Timestamp("2023-08-18 12:00"), "fhr": 0, "step": 0,
                          "var": "smoke", "value": 10.})
    cache.attrs.update(hrrr_request={"points": point_hash}, aqm_points=point_hash)
    for module in (aqm, hrrr):
        monkeypatch.setattr(module, "load_cycles", lambda *args, **kwargs: cache.copy())
    calls = []

    def forecast(study, day, source):
        calls.append((study.name, day, source))
        return forecasts.copy()

    monkeypatch.setattr(infer, "forecast_day", forecast)
    return config, cell_path, run, output, registry, calls


def test_neighborhood_export_retains_forecast_cells_and_support(prepared_neighborhood, forecasts):
    config, cells, run, output, _registry, calls = prepared_neighborhood
    forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    assert calls == [("synthetic", date(2023, 8, 18), run)]
    pd.testing.assert_frame_equal(pd.read_parquet(output / "forecast.parquet"), forecasts)
    assert gpd.read_parquet(output / "cells.parquet").point_id.to_list() == ["a", "b", "c"]
    assert len(pd.read_parquet(output / "grid_support.parquet")) == 3
    summary = json.loads((output / "summary.json").read_text())
    assert summary["issue_date"] == "2023-08-18"
    assert summary["model_members"] == 5
    inputs = pd.read_parquet(output / "input_coverage.parquet")
    assert set(inputs.point_id) == {"a", "b", "c"}
    assert set(inputs.source) == {"hrrr", "aqm"}
    assert inputs.status.eq("missing_rows").all()
    assert inputs.history_back.eq(0).all()
    assert sum(row["n_points"] for row in summary["input_coverage"]) == len(inputs)
    assert (output / "map.png").read_bytes().startswith(b"\x89PNG")


def test_neighborhood_export_uses_committed_study_api(prepared_neighborhood, monkeypatch):
    from serein.config import Study

    monkeypatch.delattr(Study, "in_study", raising=False)
    config, cells, run, output, _registry, calls = prepared_neighborhood
    forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    assert len(calls) == 1
    assert (output / "forecast.parquet").exists()


@pytest.mark.parametrize("day", [date(2024, 8, 18), date(2025, 8, 18), date(2023, 1, 1)])
def test_neighborhood_export_refuses_dates_outside_development(prepared_neighborhood, day):
    config, cells, run, output, _registry, calls = prepared_neighborhood
    with pytest.raises(ValueError, match="calibration season"):
        forecast_neighborhood(config, day, run, cells, output)
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize("missing", ["aqm", "hrrr", "registry"])
def test_neighborhood_requires_prepared_grid_before_inference(prepared_neighborhood, monkeypatch, missing):
    from serein.data import aqm, hrrr, points

    config, cells, run, output, registry, calls = prepared_neighborhood
    if missing == "registry":
        monkeypatch.setattr(points, "load_points", lambda name: registry.iloc[1:].copy())
    else:
        module = aqm if missing == "aqm" else hrrr
        monkeypatch.setattr(module, "load_cycles", lambda *args, **kwargs: pd.DataFrame({"point_id": ["a"]}))
    with pytest.raises(ValueError, match="grid"):
        forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    assert not calls
    assert not output.exists()


def test_neighborhood_refuses_existing_export_before_inference(prepared_neighborhood):
    config, cells, run, output, _registry, calls = prepared_neighborhood
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    assert not calls
    assert sentinel.read_text() == "original"


@pytest.mark.parametrize("source", ["aqm", "hrrr"])
def test_neighborhood_rejects_stale_extraction_metadata(prepared_neighborhood, monkeypatch, source):
    from serein.data import aqm, hrrr

    config, cells, run, output, registry, calls = prepared_neighborhood
    cache = pd.DataFrame({"point_id": registry.point_id})
    cache.attrs.update(hrrr_request={"points": "stale"}, aqm_points="stale")
    monkeypatch.setattr(aqm if source == "aqm" else hrrr, "load_cycles", lambda *args, **kwargs: cache.copy())
    with pytest.raises(ValueError, match="metadata must match"):
        forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    assert not calls
    assert not output.exists()


def test_neighborhood_honors_study_exclusions(prepared_neighborhood):
    config, cells, run, output, _registry, calls = prepared_neighborhood
    study = yaml.safe_load(config.read_text())
    study["exclude_regions"] = [study["region"]]
    config.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match="exclusion rules"):
        forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    assert not calls
    assert not output.exists()


def test_neighborhood_checks_noaa_inputs_for_history_days(prepared_neighborhood, monkeypatch):
    from serein.data import hrrr

    config, cells, run, output, _registry, calls = prepared_neighborhood
    metadata = json.loads((run / "model.json").read_text())
    metadata["history"] = 4
    (run / "model.json").write_text(json.dumps(metadata))
    original = hrrr.load_cycles
    seen = []

    def load(days, **kwargs):
        seen.extend(days)
        return original(days, **kwargs) if days == [date(2023, 8, 18)] else pd.DataFrame()

    monkeypatch.setattr(hrrr, "load_cycles", load)
    with pytest.raises(ValueError, match="2023-08-17"):
        forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    assert seen == [date(2023, 8, 18), date(2023, 8, 17)]
    assert not calls
    assert not output.exists()


def test_neighborhood_audits_each_history_day(prepared_neighborhood, monkeypatch):
    from serein.data import aqm, hrrr

    config, cells, run, output, _registry, calls = prepared_neighborhood
    metadata = json.loads((run / "model.json").read_text())
    metadata["history"] = 2
    (run / "model.json").write_text(json.dumps(metadata))
    for module in (hrrr, aqm):
        original = module.load_cycles

        def load(days, _original=original, **kwargs):
            frame = _original(days, **kwargs)
            frame["issue_date"] = days[0]
            frame["valid_utc"] = pd.Timestamp(days[0]) + pd.Timedelta(hours=12)
            return frame

        monkeypatch.setattr(module, "load_cycles", load)
    forecast_neighborhood(config, date(2023, 8, 18), run, cells, output)
    report = pd.read_parquet(output / "input_coverage.parquet")
    assert len(calls) == 1
    assert set(report.history_back) == {0, 1}
    for back, group in report.groupby("history_back"):
        assert group.issue_date.eq(date(2023, 8, 18) - pd.Timedelta(days=back)).all()
        assert set(group.source) == {"hrrr", "aqm"}
        assert set(group.point_id) == {"a", "b", "c"}
