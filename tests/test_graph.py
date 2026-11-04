import numpy as np
import pytest

from serein.features.graph import build_knn_graph, idw_neighbor_mean


@pytest.mark.parametrize("lat, lon, k, max_km", [
    ([], [], 10, 300.0),
    ([38.0], [-122.0], 10, 300.0),
    ([38.0, 39.0], [-122.0, -122.0], 0, 300.0),
    ([38.0, 39.0], [-122.0, -122.0], 10, 1.0),
])
def test_graph_without_neighbors(lat, lon, k, max_km):
    graph = build_knn_graph(lat, lon, k=k, max_km=max_km)
    assert graph.n_edges == 0
    assert graph.src.shape == graph.dst.shape == (0,)
    assert graph.src.dtype == graph.dst.dtype == np.int64
    assert graph.dist_km.shape == graph.bearing.shape == (0,)
    assert graph.edge_attr.shape == (0, 4)
    assert graph.edge_attr.dtype == np.float32
    values = np.ones((2, len(lat)))
    means = idw_neighbor_mean(values, graph)
    assert means.shape == values.shape
    assert np.isnan(means).all()


@pytest.mark.parametrize("k", [-1, 1.5])
def test_graph_rejects_invalid_neighbor_count(k):
    with pytest.raises(ValueError, match="k must be a nonnegative integer"):
        build_knn_graph([38.0, 38.0], [-122.0, -122.0], k=k)


@pytest.mark.parametrize("max_km", [0.0, -1.0, np.nan, np.inf])
def test_graph_rejects_invalid_distance_limit(max_km):
    with pytest.raises(ValueError, match="max_km must be finite and positive"):
        build_knn_graph([38.0, 38.0], [-122.0, -122.0], max_km=max_km)


@pytest.mark.parametrize("lat, lon", [
    ([91.0], [0.0]), ([-91.0], [0.0]),
    ([0.0], [181.0]), ([0.0], [-181.0]),
    ([np.nan], [0.0]), ([0.0], [np.inf]),
    ([38.0, 39.0], [-122.0]),
    ([[38.0]], [[-122.0]]), (38.0, -122.0),
])
def test_graph_rejects_invalid_coordinates(lat, lon):
    with pytest.raises(ValueError, match="coordinates"):
        build_knn_graph(lat, lon)


def test_graph_accepts_geographic_boundaries_and_links_across_dateline():
    graph = build_knn_graph([0.0, 0.0, -90.0, 90.0],
                            [-179.9, 179.9, -180.0, 180.0], k=1, max_km=30.0)
    assert set(zip(graph.src, graph.dst)) == {(0, 1), (1, 0)}
    assert np.all((graph.dist_km > 20) & (graph.dist_km < 25))
