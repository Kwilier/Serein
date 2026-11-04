import numpy as np
import pytest

from serein.eval.metrics import (
    bootstrap_rmse_diff,
    crps_from_quantiles,
    interval_metrics,
    point_metrics,
)


@pytest.mark.parametrize("pred,obs", [
    (np.ones((2, 1)), np.ones(2)),
    (np.ones((2, 1)), np.ones((2, 1))),
    (np.ones(2), np.ones(3)),
])
def test_point_metrics_requires_matching_vectors(pred, obs):
    with pytest.raises(ValueError, match="shape|one-dimensional"):
        point_metrics(pred, obs)


@pytest.mark.parametrize("dates", [np.arange(3), np.ones((2, 1))])
def test_point_metrics_validates_dates_even_without_finite_predictions(dates):
    with pytest.raises(ValueError, match="dates"):
        point_metrics(np.full(2, np.nan), np.ones(2), dates)


@pytest.mark.parametrize("bad_index", range(4))
def test_bootstrap_requires_matching_vectors(bad_index):
    inputs = [np.ones(6), np.ones(6), np.ones(6), np.arange(6)]
    inputs[bad_index] = inputs[bad_index][:, None]
    with pytest.raises(ValueError, match="shape|one-dimensional"):
        bootstrap_rmse_diff(*inputs)


@pytest.mark.parametrize("qs,levels,obs", [
    (np.ones((2, 3)), [0.1, 0.5, 0.9], np.ones((2, 1))),
    (np.ones((2, 3)), [0.1, 0.5, 0.9], np.ones(1)),
    (np.ones((2, 3)), [0.5], np.ones(2)),
    (np.ones(2), [0.5], np.ones(2)),
    (np.ones((2, 0)), [], np.ones(2)),
])
def test_crps_rejects_shapes_that_would_broadcast(qs, levels, obs):
    with pytest.raises(ValueError, match="shape|quantile"):
        crps_from_quantiles(qs, levels, obs)


def test_scoring_accepts_lists_and_preserves_row_order():
    assert point_metrics([2., 5.], [1., 3.])["mae"] == 1.5
    np.testing.assert_allclose(
        crps_from_quantiles([[2., 2., 2.], [5., 5., 5.]], [0.1, 0.5, 0.9], [1., 3.]),
        [1., 2.],
    )
    assert crps_from_quantiles(np.empty((0, 3)), [0.1, 0.5, 0.9], []).shape == (0,)


def test_interval_metrics_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="lower.*upper"):
        interval_metrics(np.array([5., 20.]), np.array([15., 10.]), np.array([10., 15.]))


@pytest.mark.parametrize("lo,hi,obs", [
    (np.zeros((2, 1)), np.ones(2), np.ones(2)),
    (np.zeros(2), np.ones((2, 1)), np.ones(2)),
    (np.zeros(2), np.ones(2), np.ones((2, 1))),
])
def test_interval_metrics_rejects_mismatched_shapes(lo, hi, obs):
    with pytest.raises(ValueError, match="shape"):
        interval_metrics(lo, hi, obs)


def test_interval_metrics_keeps_boundaries_and_ignores_missing_rows():
    result = interval_metrics(
        np.array([5., 10., 20., np.nan, 0., 30.]),
        np.array([15., 10., 25., 10., np.inf, 20.]),
        np.array([5., 10., 30., 5., 5., np.nan]),
    )
    assert result == {"coverage90": pytest.approx(2 / 3), "width90": 5.}


def test_interval_metrics_without_complete_rows():
    assert interval_metrics(np.array([np.nan]), np.array([10.]), np.array([5.])) == {}
    assert interval_metrics(np.array([]), np.array([]), np.array([])) == {}


@pytest.mark.parametrize("levels", [
    [-0.1, 0.5, 0.9], [0.1, 0.5, 1.1], [0.1, np.nan, 0.9],
    [0.1, 0.5, np.inf], [0.1, 0.1, 0.9], [0.9, 0.5, 0.1],
])
def test_crps_rejects_invalid_probability_levels(levels):
    with pytest.raises(ValueError, match="levels"):
        crps_from_quantiles([[1., 2., 3.]], levels, [2.])


def test_crps_rejects_crossed_quantiles():
    with pytest.raises(ValueError, match="quantiles.*decreas"):
        crps_from_quantiles([[1., 3., 2.]], [0.1, 0.5, 0.9], [2.])


def test_crps_accepts_probability_endpoints_and_tied_quantiles():
    np.testing.assert_allclose(
        crps_from_quantiles([[2., 2., 2.]], [0., 0.5, 1.], [5.]), [3.],
    )
    np.testing.assert_allclose(crps_from_quantiles([[2.]], [0.5], [5.]), [3.])
