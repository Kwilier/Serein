import numpy as np
import pytest

from serein.models.conformal import CQR, calibrate_quantiles
from serein.models.gnn import QUANTILES


def test_shrinking_interval_preserves_quantile_order_and_median():
    raw = np.array([[[1., 2., 3., 4., 5.]]])
    original = raw.copy()
    calibration = CQR(qhat={(0, -1): -1.5})

    actual = calibrate_quantiles(raw, calibration, QUANTILES)

    np.testing.assert_allclose(actual, [[[2.5, 2.5, 3., 3.5, 3.5]]])
    assert (np.diff(actual, axis=-1) >= 0).all()
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("levels", [QUANTILES, (0.05, 0.5, 0.95)])
def test_calibrated_pollution_has_nonnegative_support(levels):
    raw = np.linspace(-1, 1, len(levels))[None, None, :]
    actual = calibrate_quantiles(raw, CQR(qhat={(0, -1): 2.0}), levels)
    assert (np.expm1(actual) >= 0).all()
    assert (np.diff(actual, axis=-1) >= 0).all()
    assert actual[0, 0, len(levels) // 2] == 0


def test_calibration_keeps_missing_rows_and_supports_empty_batches():
    raw = np.array([[[1., 2., 3., 4., 5.]], [[np.nan] * 5]])
    actual = calibrate_quantiles(raw, CQR(), QUANTILES)
    np.testing.assert_array_equal(actual[0], raw[0])
    assert np.isnan(actual[1]).all()
    assert calibrate_quantiles(np.empty((0, 2, 5)), CQR(), QUANTILES).shape == (0, 2, 5)


def test_over_shrinking_interval_collapses_at_the_median():
    raw = np.array([[[1., 2., 3., 4., 5.]]])
    actual = calibrate_quantiles(raw, CQR(qhat={(0, -1): -10.0}), QUANTILES)
    np.testing.assert_array_equal(actual, np.full_like(raw, 3.0))


@pytest.mark.parametrize("raw,levels", [
    (np.zeros((2, 5)), QUANTILES),
    (np.zeros((2, 1, 3)), QUANTILES),
    (np.zeros((2, 1, 3)), (0.05, 0.4, 0.95)),
    (np.zeros((2, 1, 3)), (0.5, 0.05, 0.95)),
    (np.zeros((2, 1, 3)), (0.5, 0.75, 0.95)),
    (np.array([[[1., 3., 2.]]]), (0.05, 0.5, 0.95)),
    (np.array([[[1., 2., np.inf]]]), (0.05, 0.5, 0.95)),
])
def test_invalid_forecast_distributions_fail_clearly(raw, levels):
    with pytest.raises(ValueError, match="quantile"):
        calibrate_quantiles(raw, CQR(), levels)
