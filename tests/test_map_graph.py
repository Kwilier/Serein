import numpy as np
import pytest

from serein.models.infer import sensor_plus_grid_graph


@pytest.mark.parametrize("n_sensors", [0, 1, 2])
@pytest.mark.parametrize("n_grid", [0, 1, 3])
@pytest.mark.parametrize("k", [0, 1, 10])
def test_sparse_map_graph_uses_only_available_sensor_neighbors(n_sensors, n_grid, k):
    sensors = np.column_stack([np.full(n_sensors, 38.), -122. + np.arange(n_sensors) * .01])
    grid = np.column_stack([np.full(n_grid, 38.02), -122. + np.arange(n_grid) * .01])
    graph = sensor_plus_grid_graph(sensors, grid, k=k, max_km=100)

    assert graph.edge_attr.shape == (graph.n_edges, 4)
    assert np.isfinite(graph.edge_attr).all()
    assert (graph.src < n_sensors).all()
    assert (graph.dst < n_sensors + n_grid).all()
    assert (graph.src != graph.dst).all()
    expected_grid_edges = n_grid * min(k, n_sensors)
    assert np.count_nonzero(graph.dst >= n_sensors) == expected_grid_edges


def test_map_grid_does_not_listen_to_distant_sensors():
    graph = sensor_plus_grid_graph([[38., -122.]], [[45., -100.]], k=10, max_km=10)
    assert graph.n_edges == 0


@pytest.mark.parametrize("grid", [[], [[91., 0.]], [[0., 181.]], [[np.nan, 0.]], [[0.]]])
def test_map_graph_rejects_invalid_coordinate_shapes_and_values(grid):
    with pytest.raises(ValueError, match="coordinates"):
        sensor_plus_grid_graph([[38., -122.]], grid, k=1, max_km=100)
