import numpy as np
import pytest

from serein.eval.metrics import interval_metrics


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
