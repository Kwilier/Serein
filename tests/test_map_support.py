import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from serein.data.firms import haversine_km
from serein.eval.map_support import (
    audit_forecasts,
    main,
    plot_support,
    reading_support,
    sensor_support,
)


@pytest.fixture
def points():
    return pd.DataFrame({"point_id": ["A", "B", "untrained", "near", "far"],
                         "kind": ["epa", "purpleair", "epa", "grid", "grid"],
                         "lat": [38., 38., 38., 38., 45.],
                         "lon": [-122., -121.9, -100., -121.95, -100.]})


@pytest.fixture
def forecasts():
    return pd.DataFrame({"point_id": ["near", "far", "A"], "kind": ["grid", "grid", "epa"],
                         "issue_date": "2023-08-18", "lead": 1, "q50": [12., 25., 15.]})


def test_support_uses_trained_available_sensors_and_retains_distant_cells(points):
    result, metadata = sensor_support(points, ["A", "B", "missing"], k=10, max_km=100)
    result = result.set_index("point_id")
    assert result.loc["near", "n_sensor_neighbors"] == 2
    assert result.loc["near", "sensor_support"] == "connected"
    assert result.loc["near", "closest_used_sensor_km"] == pytest.approx(
        haversine_km(38., -121.95, 38., -122.))
    assert result.loc["far", "n_sensor_neighbors"] == 0
    assert result.loc["far", "sensor_support"] == "no_sensor_edges"
    assert np.isnan(result.loc["far", "closest_used_sensor_km"])
    assert np.isnan(result.loc["far", "farthest_used_sensor_km"])
    assert result.loc["far", "nearest_trained_sensor_km"] > 100
    assert metadata["n_available_trained_sensors"] == 2
    assert metadata["n_trained_sensors"] == 3
    assert metadata["missing_trained_sensor_ids"] == ["missing"]
    assert metadata["n_grid_without_sensor_edges"] == 1


def test_support_respects_neighbor_count_and_inclusive_radius(points):
    grid = points.loc[points.point_id.eq("near")].iloc[0]
    radius = float(haversine_km(grid.lat, grid.lon, 38., -122.))
    support, _ = sensor_support(points, ["A"], k=1, max_km=radius)
    assert support.iloc[0].n_sensor_neighbors == 1
    support, _ = sensor_support(points, ["A"], k=1, max_km=radius / 2)
    assert support.iloc[0].n_sensor_neighbors == 0


def test_zero_neighbors_and_no_available_sensors_are_auditable(points):
    support, metadata = sensor_support(points, ["missing"], k=10, max_km=100)
    assert support.n_sensor_neighbors.eq(0).all()
    assert support.nearest_trained_sensor_km.isna().all()
    assert metadata["n_available_trained_sensors"] == 0
    support, _ = sensor_support(points, ["A"], k=0, max_km=100)
    assert support.n_sensor_neighbors.eq(0).all()
    assert support.nearest_trained_sensor_km.notna().all()


def test_untrained_sensor_at_grid_location_cannot_supply_an_edge(points):
    points.loc[points.point_id.eq("untrained"), ["lat", "lon"]] = [45., -100.]
    support, _ = sensor_support(points, ["A", "B"], k=10, max_km=100)
    far = support.set_index("point_id").loc["far"]
    assert far.n_sensor_neighbors == 0
    assert far.nearest_trained_sensor_km > 100


def test_nearest_sensor_uses_spherical_distance_across_dateline():
    points = pd.DataFrame({"point_id": ["west", "east", "grid"], "kind": ["epa", "epa", "grid"],
                           "lat": [0., 0., 0.], "lon": [179.9, -170., -179.9]})
    support, _ = sensor_support(points, ["west", "east"], k=1, max_km=100)
    assert support.iloc[0].n_sensor_neighbors == 1
    assert support.iloc[0].nearest_trained_sensor_km == pytest.approx(haversine_km(0, 179.9, 0, -179.9))


def test_support_is_independent_of_registry_row_order(points):
    expected, meta = sensor_support(points, ["B", "A"], k=1, max_km=100)
    actual, actual_meta = sensor_support(points.sample(frac=1, random_state=3), ["B", "A"], k=1, max_km=100)
    pd.testing.assert_frame_equal(expected.set_index("point_id").sort_index(),
                                   actual.set_index("point_id").sort_index())
    assert meta == actual_meta


@pytest.mark.parametrize("nodes", [None, [], "A", [None], [""], ["A", "A"], [1]])
def test_invalid_trained_nodes_are_rejected(points, nodes):
    with pytest.raises(ValueError, match="node_ids"):
        sensor_support(points, nodes, k=1, max_km=100)


@pytest.mark.parametrize("k,radius", [(-1, 100), (True, 100), (1.5, 100), (1, 0),
                                      (1, np.nan), (1, np.inf), (1, True)])
def test_invalid_graph_options_are_rejected(points, k, radius):
    with pytest.raises(ValueError):
        sensor_support(points, ["A"], k=k, max_km=radius)


@pytest.mark.parametrize("problem", ["duplicate", "missing_id", "missing_lat", "invalid_lat",
                                      "invalid_lon", "no_grid", "trained_grid"])
def test_invalid_registry_is_rejected(points, problem):
    nodes = ["A"]
    if problem == "duplicate":
        points.loc[1, "point_id"] = "A"
    elif problem == "missing_id":
        points.loc[0, "point_id"] = None
    elif problem == "missing_lat":
        points = points.drop(columns="lat")
    elif problem == "invalid_lat":
        points.loc[0, "lat"] = 91
    elif problem == "invalid_lon":
        points.loc[0, "lon"] = np.nan
    elif problem == "no_grid":
        points = points[points.kind != "grid"]
    else:
        nodes = ["near"]
    with pytest.raises(ValueError):
        sensor_support(points, nodes, k=1, max_km=100)


def test_forecast_audit_keeps_groups_and_flags_missing_or_invalid_medians(points, forecasts):
    support, _ = sensor_support(points, ["A", "B"], k=2, max_km=100)
    later = forecasts.copy()
    later["issue_date"] = "2023-08-19"
    later["q50"] = [-1, np.nan, np.inf]
    result = audit_forecasts(pd.concat([forecasts, later]), support)
    assert len(result) == 4
    assert set(result.point_id) == {"near", "far"}
    assert result.valid_median.tolist() == [True, True, False, False]
    assert result.n_sensor_neighbors.tolist() == [2, 0, 2, 0]
    assert result.q50.iloc[2] == -1
    assert np.isnan(result.q50.iloc[3])


def test_audit_preserves_nullable_missing_median(points, forecasts):
    support, _ = sensor_support(points, ["A"], k=1, max_km=100)
    forecasts["q50"] = forecasts.q50.astype("Float64")
    forecasts.loc[0, "q50"] = pd.NA
    result = audit_forecasts(forecasts, support)
    assert not result.iloc[0].valid_median
    assert pd.isna(result.iloc[0].q50)


@pytest.mark.parametrize("problem", ["unknown", "duplicate", "missing_date", "coordinate_mismatch",
                                      "partial_coordinates", "no_grid", "collision"])
def test_inconsistent_saved_forecasts_are_rejected(points, forecasts, problem):
    support, _ = sensor_support(points, ["A"], k=1, max_km=100)
    if problem == "unknown":
        forecasts.loc[0, "point_id"] = "absent"
    elif problem == "duplicate":
        forecasts = pd.concat([forecasts, forecasts.iloc[[0]]])
    elif problem == "missing_date":
        forecasts.loc[0, "issue_date"] = None
    elif problem == "coordinate_mismatch":
        forecasts["lat"], forecasts["lon"] = 0., 0.
    elif problem == "partial_coordinates":
        forecasts["lat"] = 38.
    elif problem == "no_grid":
        forecasts = forecasts[forecasts.kind != "grid"]
    else:
        forecasts["n_sensor_neighbors"] = 100
    with pytest.raises(ValueError):
        audit_forecasts(forecasts, support)


def test_offline_plot_handles_no_available_sensors(tmp_path, points):
    support, _ = sensor_support(points, ["missing"], k=1, max_km=100)
    output = tmp_path / "support.png"
    plot_support(support, points.iloc[:0], output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        plot_support(support, points.iloc[:0], output)
    assert output.read_bytes() == original


def test_cli_writes_auditable_report_without_model_weights(tmp_path, points, forecasts):
    config, model, registry, forecast = [tmp_path / name for name in
                                       ["study.yaml", "model.json", "points.parquet", "forecast.parquet"]]
    config.write_text(yaml.safe_dump({"name": "synthetic", "graph": {"k_neighbors": 2, "max_km": 100}}))
    model.write_text(json.dumps({"node_ids": ["A", "B", "missing"]}))
    points.to_parquet(registry)
    forecasts.to_parquet(forecast)
    output = tmp_path / "audit"
    assert main(["--config", str(config), "--model-json", str(model), "--points", str(registry),
                 "--forecasts", str(forecast), "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["missing_trained_sensor_ids"] == ["missing"]
    assert summary["sources"]["forecasts"] == str(forecast.resolve())
    assert summary["source_sha256"]["forecasts"] == hashlib.sha256(forecast.read_bytes()).hexdigest()
    audit = pd.read_parquet(output / "forecast_audit.parquet")
    assert len(audit) == 2
    counts = pd.read_csv(output / "forecast_counts.csv")
    assert counts.iloc[0].n_grid == 2
    assert counts.iloc[0].n_without_sensor_edges == 1
    assert counts.iloc[0].n_invalid_medians == 0
    assert (output / "support.png").read_bytes().startswith(b"\x89PNG")


def test_cli_refuses_existing_report_before_reading_inputs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(SystemExit) as error:
        main(["--config", "absent.yaml", "--model-json", "absent.json", "--points", "absent.parquet",
              "--forecasts", "absent.parquet", "--output", str(output)])
    assert error.value.code == 2
    assert sentinel.read_text() == "original"


def test_reading_support_distinguishes_missing_records_readings_and_edges(points, forecasts):
    support, _ = sensor_support(points, ["A", "B"], k=2, max_km=100)
    days = []
    for day in ["2023-08-18", "2023-08-19", "2023-08-20"]:
        frame = forecasts.copy()
        frame["issue_date"] = day
        days.append(frame)
    audit = audit_forecasts(pd.concat(days, ignore_index=True), support)
    features = pd.DataFrame({"point_id": ["A", "B", "A"],
                             "issue_date": ["2023-08-18", "2023-08-18", "2023-08-19"],
                             "obs_last24": [10., np.nan, np.nan], "obs_prevday": [20., 30., np.nan]})
    result = reading_support(audit, points, ["A", "B"], features, k=2, max_km=100)
    near = result.loc[result.point_id.eq("near")]
    assert near.n_neighbors_with_feature_row.tolist() == [2, 1, 0]
    assert near.n_neighbors_with_last24.tolist() == [1, 0, 0]
    assert near.n_neighbors_with_prevday.tolist() == [2, 0, 0]
    assert near.reading_support.tolist() == ["last24_available", "no_last24_readings", "no_feature_records"]
    far = result.loc[result.point_id.eq("far")]
    assert far.n_neighbors_with_feature_row.eq(0).all()
    assert far.reading_support.eq("no_sensor_edges").all()


@pytest.mark.parametrize("problem", ["duplicate", "time", "timezone", "missing_date", "missing_column",
                                      "missing_id", "nonnumeric", "graph_mismatch"])
def test_reading_support_rejects_inconsistent_feature_tables(points, forecasts, problem):
    support, _ = sensor_support(points, ["A", "B"], k=2, max_km=100)
    audit = audit_forecasts(forecasts, support)
    features = pd.DataFrame({"point_id": ["A"], "issue_date": ["2023-08-18"],
                             "obs_last24": [10.], "obs_prevday": [20.]})
    k = 2
    if problem == "duplicate":
        features = pd.concat([features, features])
    elif problem == "time":
        features["issue_date"] = "2023-08-18 01:00:00"
    elif problem == "timezone":
        features["issue_date"] = "2023-08-18T00:00:00Z"
    elif problem == "missing_date":
        features["issue_date"] = None
    elif problem == "missing_column":
        features = features.drop(columns="obs_last24")
    elif problem == "missing_id":
        features["point_id"] = None
    elif problem == "nonnumeric":
        features["obs_last24"] = "invalid"
    else:
        k = 1
    with pytest.raises(ValueError):
        reading_support(audit, points, ["A", "B"], features, k=k, max_km=100)


def test_reading_support_excludes_untrained_records_and_infinite_readings(points, forecasts):
    support, _ = sensor_support(points, ["A", "B"], k=2, max_km=100)
    audit = audit_forecasts(forecasts, support)
    features = pd.DataFrame({"point_id": ["A", "B", "untrained"], "issue_date": "2023-08-18",
                             "obs_last24": [np.inf, -1., 100.], "obs_prevday": [np.nan, 2., 100.]})
    result = reading_support(audit, points, ["A", "B"], features, k=2, max_km=100)
    assert result.iloc[0].n_neighbors_with_feature_row == 2
    assert result.iloc[0].n_neighbors_with_last24 == 1
    assert result.iloc[0].n_neighbors_with_prevday == 1


def test_cli_can_join_prepared_reading_records(tmp_path, points, forecasts):
    config, model, registry, forecast, feature = [tmp_path / name for name in
        ["study.yaml", "model.json", "points.parquet", "forecast.parquet", "features.parquet"]]
    config.write_text(yaml.safe_dump({"name": "synthetic", "graph": {"k_neighbors": 2, "max_km": 100}}))
    model.write_text(json.dumps({"node_ids": ["A", "B"]}))
    points.to_parquet(registry)
    forecasts.to_parquet(forecast)
    pd.DataFrame({"point_id": ["A", "B"], "issue_date": "2023-08-18",
                  "obs_last24": [10., np.nan], "obs_prevday": [20., 30.]}).to_parquet(feature)
    output = tmp_path / "audit"
    assert main(["--config", str(config), "--model-json", str(model), "--points", str(registry),
                 "--forecasts", str(forecast), "--features", str(feature), "--output", str(output)]) == 0
    result = pd.read_parquet(output / "forecast_audit.parquet")
    assert result.iloc[0].n_neighbors_with_last24 == 1
    counts = pd.read_csv(output / "forecast_counts.csv")
    assert counts.iloc[0].n_without_last24_readings == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["sources"]["features"] == str(feature.resolve())
