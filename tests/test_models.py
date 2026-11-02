import numpy as np
import pytest
import torch

from serein.eval.metrics import bootstrap_rmse_diff, crps_from_quantiles, point_metrics
from serein.features.graph import build_knn_graph, idw_neighbor_mean
from serein.models.conformal import CQR
from serein.models.gnn import QUANTILES, SmokeCastGNN, pinball_loss


def _line_graph(n=12):
    lat = np.full(n, 38.0)
    lon = -122.0 + 0.3 * np.arange(n)  # west -> east, ~26 km apart
    return lat, lon, build_knn_graph(lat, lon, k=2, max_km=100)


def test_knn_graph_bearings():
    _lat, _lon, g = _line_graph()
    # Edge from node 0 (west) to node 1 (east) points east: bearing ~ +90 degrees.
    e = np.where((g.src == 0) & (g.dst == 1))[0][0]
    assert np.degrees(g.bearing[e]) == pytest.approx(90, abs=1)
    assert (g.dist_km < 100).all()


def test_idw_neighbor_mean_ignores_nan():
    _, _, g = _line_graph(3)
    v = np.array([[1.0, np.nan, 3.0]])
    nb = idw_neighbor_mean(v, g)
    assert nb[0, 1] == pytest.approx(2.0)


def test_gnn_quantiles_never_cross():
    lat, _lon, g = _line_graph()
    m = SmokeCastGNN(n_feat=5, n_leads=4, hidden=32)
    x = torch.randn(3, len(lat), 5)
    q = m(x, torch.randn(3, len(lat), 2), torch.zeros(3, len(lat), 4),
          torch.as_tensor(g.src), torch.as_tensor(g.dst),
          torch.as_tensor(g.bearing, dtype=torch.float32), torch.as_tensor(g.edge_attr))
    assert q.shape == (3, len(lat), 4, len(QUANTILES))
    assert (q.diff(dim=-1) >= 0).all()


def test_gnn_learns_from_upwind_neighbor_but_mlp_cannot():
    """Toy check of the core idea: tomorrow's smoke at a site equals TODAY's
    reading at its upwind (western) neighbor. Only the graph model can see it."""
    torch.manual_seed(0)
    lat, _lon, g = _line_graph(10)
    N, T = len(lat), 256
    obs = torch.rand(T, N, 1) * 3
    y = torch.full((T, N, 1), float("nan"))
    y[:, 1:, 0] = obs[:, :-1, 0]          # value comes from the western neighbor
    wind = torch.zeros(T, N, 2)
    wind[..., 0] = 5.0                      # wind blowing toward the east
    args = (torch.as_tensor(g.src), torch.as_tensor(g.dst),
            torch.as_tensor(g.bearing, dtype=torch.float32), torch.as_tensor(g.edge_attr))
    q = torch.tensor(QUANTILES)
    losses = {}
    for use_graph in (True, False):
        torch.manual_seed(0)
        m = SmokeCastGNN(1, 1, hidden=32, layers=2, drop=0.0, use_graph=use_graph)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(300):
            loss = pinball_loss(m(obs, wind, torch.zeros(T, N, 1), *args), y, q)
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses[use_graph] = loss.item()
    assert losses[True] < 0.5 * losses[False]


def test_conformal_reaches_target_coverage():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(0, 2, (n, 1))
    lo, hi, med = np.full((n, 1), -1.0), np.full((n, 1), 1.0), np.zeros((n, 1))  # too narrow
    cqr = CQR(alpha=0.1, bins=(10.0,)).fit(lo[:2000], hi[:2000], med[:2000], y[:2000])
    lo2, hi2 = cqr.apply(lo[2000:], hi[2000:], med[2000:])
    cov = np.mean((y[2000:] >= lo2) & (y[2000:] <= hi2))
    assert 0.87 < cov < 0.93


def test_metrics_basic():
    obs = np.array([5, 10, 40, 80.0])
    m = point_metrics(obs + 1, obs)
    assert m["bias"] == pytest.approx(1) and m["usg_pod"] == 1.0
    crps = crps_from_quantiles(np.repeat(obs[:, None], 3, 1), (0.05, 0.5, 0.95), obs)
    assert np.allclose(crps, 0)


def test_bootstrap_prefers_better_model():
    rng = np.random.default_rng(1)
    obs = rng.gamma(2, 5, 2000)
    dates = np.repeat(np.arange(100), 20)
    good = obs + rng.normal(0, 1, 2000)
    bad = obs + rng.normal(0, 5, 2000)
    r = bootstrap_rmse_diff(good, bad, obs, dates, n_boot=300)
    assert r["ci_hi"] < 0 and r["p_a_better"] > 0.99
