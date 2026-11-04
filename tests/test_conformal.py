import numpy as np

from serein.models.conformal import CQR, _qhat


def test_calibration_uses_finite_sample_order_statistic():
    scores = np.arange(100, dtype=float)
    # For 90% coverage, ceil(101 * 0.9) selects the 91st score.
    assert _qhat(scores, 0.1) == 90.0


def test_calibration_rank_ignores_missing_scores_and_caps_at_sample_size():
    scores = np.concatenate([np.arange(100, dtype=float), [np.nan, np.inf]])
    assert _qhat(scores, 0.1) == 90.0
    assert _qhat(np.array([2.0, 5.0]), 0.1) == 5.0
    assert _qhat(np.array([2.0, 5.0]), 0.0) == 5.0
    assert _qhat(np.array([2.0, 5.0]), 1.0) == 2.0


def test_calibration_applies_rank_correction_to_pooled_and_group_intervals():
    y = np.arange(100, dtype=float)[:, None]
    lo = np.zeros_like(y)
    hi = np.zeros_like(y)
    median = np.zeros_like(y)
    model = CQR().fit(lo, hi, median, y)
    assert model.qhat[(0, -1)] == 90.0
    assert model.qhat[(0, 0)] == 90.0
    lower, upper = model.apply(lo[:1], hi[:1], median[:1])
    np.testing.assert_array_equal(lower, [[-90.0]])
    np.testing.assert_array_equal(upper, [[90.0]])


def test_refit_discards_group_corrections_from_previous_sample():
    median = np.zeros((60, 1))
    lo = np.zeros_like(median)
    hi = np.ones_like(median)
    reused = CQR(bins=(10.0,)).fit(lo, hi, median, np.full_like(median, 8.0))
    # The smaller new sample must use its pooled correction, not the old group.
    new_y = np.full((10, 1), 3.0)
    reused.fit(lo[:10], hi[:10], median[:10], new_y)
    fresh = CQR(bins=(10.0,)).fit(lo[:10], hi[:10], median[:10], new_y)
    for actual, expected in zip(reused.apply(lo, hi, median), fresh.apply(lo, hi, median)):
        np.testing.assert_array_equal(actual, expected)


def test_refit_discards_leads_absent_from_new_sample():
    median = np.zeros((60, 2))
    lo = np.zeros_like(median)
    hi = np.ones_like(median)
    model = CQR().fit(lo, hi, median, np.full_like(median, 8.0))
    model.fit(lo[:, :1], hi[:, :1], median[:, :1], np.full((60, 1), 3.0))
    assert all(lead == 0 for lead, group in model.qhat)


def test_apply_preserves_fractional_corrections_for_integer_forecasts():
    model = CQR(qhat={(0, -1): 0.75, (1, -1): -0.25})
    lo = np.array([[2, 4], [3, 5]])
    hi = lo + 2
    median = lo + 1
    lower, upper = model.apply(lo, hi, median)
    np.testing.assert_allclose(lower, [[1.25, 4.25], [2.25, 5.25]])
    np.testing.assert_allclose(upper, [[4.75, 5.75], [5.75, 6.75]])
    np.testing.assert_array_equal(lo, [[2, 4], [3, 5]])
    np.testing.assert_array_equal(hi, [[4, 6], [5, 7]])
