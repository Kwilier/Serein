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
