import numpy as np
import pytest

from serein.models.conformal import CQR, _qhat


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("shape", [(1, 2), (3, 1), (3,), (3, 2, 1)])
def test_fit_rejects_mismatched_shapes_without_changing_calibration(index, shape):
    arrays = [np.zeros((3, 2)) for _ in range(4)]
    arrays[index] = np.zeros(shape)
    model = CQR(qhat={(0, -1): 0.75})
    with pytest.raises(ValueError, match="shape"):
        model.fit(*arrays)
    assert model.qhat == {(0, -1): 0.75}


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("shape", [(1, 2), (3, 1), (3,), (3, 2, 1)])
def test_apply_rejects_mismatched_shapes(index, shape):
    arrays = [np.zeros((3, 2)) for _ in range(3)]
    arrays[index] = np.zeros(shape)
    with pytest.raises(ValueError, match="shape"):
        CQR().apply(*arrays)


def test_calibration_accepts_empty_row_batches():
    empty = np.empty((0, 2))
    model = CQR().fit(empty, empty, empty, empty)
    lower, upper = model.apply(empty, empty, empty)
    assert lower.shape == upper.shape == (0, 2)


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

@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_invalid_alpha_preserves_existing_calibration(alpha):
    model = CQR(alpha=alpha, qhat={(0, -1): 0.75})
    values = np.zeros((3, 1))
    with pytest.raises(ValueError, match="alpha"):
        model.fit(values, values, values, values)
    assert model.qhat == {(0, -1): 0.75}


@pytest.mark.parametrize("bins", [(2.0, 1.0), (1.0, 1.0), (np.nan,),
                                  (np.inf,), (-np.inf,), ((1.0, 2.0),)])
def test_invalid_bins_preserve_existing_calibration(bins):
    model = CQR(bins=bins, qhat={(0, -1): 0.75})
    values = np.zeros((3, 1))
    with pytest.raises(ValueError, match="bins"):
        model.fit(values, values, values, values)
    assert model.qhat == {(0, -1): 0.75}
    with pytest.raises(ValueError, match="bins"):
        model.apply(values, values, values)


def test_empty_bins_use_pooled_calibration():
    model = CQR(bins=())
    values = np.zeros((3, 1))
    model.fit(values, values, values, np.full_like(values, 2.0))
    lower, upper = model.apply(values, values, values)
    np.testing.assert_array_equal(lower, np.full_like(values, -2.0))
    np.testing.assert_array_equal(upper, np.full_like(values, 2.0))


def test_reversed_bounds_are_rejected_before_replacing_calibration():
    lo = np.array([[0.0], [3.0]])
    hi = np.array([[1.0], [2.0]])
    median = (lo + hi) / 2
    model = CQR(qhat={(0, -1): 0.75})
    with pytest.raises(ValueError, match="bounds"):
        model.fit(lo, hi, median, median)
    assert model.qhat == {(0, -1): 0.75}
    with pytest.raises(ValueError, match="bounds"):
        model.apply(lo, hi, median)


def test_equal_and_missing_bounds_remain_supported():
    lo = np.array([[1.0], [np.nan]])
    hi = np.array([[1.0], [np.nan]])
    median = np.ones_like(lo)
    model = CQR().fit(lo, hi, median, np.array([[2.0], [4.0]]))
    lower, upper = model.apply(lo, hi, median)
    np.testing.assert_allclose(lower, [[0.0], [np.nan]])
    np.testing.assert_allclose(upper, [[2.0], [np.nan]])


@pytest.mark.parametrize("dtype, lower, upper, observed", [
    (np.uint8, 10, 20, 5),
    (np.int8, -100, -90, 100),
])
def test_integer_calibration_matches_floating_point(dtype, lower, upper, observed):
    lo = np.full((60, 1), lower, dtype=dtype)
    hi = np.full((60, 1), upper, dtype=dtype)
    median = (lo.astype(float) + hi.astype(float)) / 2
    y = np.full((60, 1), observed, dtype=dtype)
    expected = CQR().fit(lo.astype(float), hi.astype(float), median, y.astype(float))
    actual = CQR().fit(lo, hi, median, y)
    assert actual.qhat == expected.qhat
    for result, reference in zip(actual.apply(lo, hi, median), expected.apply(lo, hi, median)):
        np.testing.assert_allclose(result, reference)


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_medians_use_pooled_correction(missing):
    model = CQR(qhat={(0, -1): 1.0, (0, 0): 3.0, (0, 2): 5.0})
    lo = np.zeros((1, 1))
    hi = np.ones((1, 1))
    lower, upper = model.apply(lo, hi, np.array([[missing]]))
    np.testing.assert_allclose(lower, [[-1.0]])
    np.testing.assert_allclose(upper, [[2.0]])


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_medians_do_not_contaminate_smoke_group(missing):
    lo = np.zeros((120, 1))
    hi = np.ones_like(lo)
    median = np.full_like(lo, 4.0)
    median[60:] = missing
    y = np.full_like(lo, 2.0)
    y[60:] = 11.0
    model = CQR().fit(lo, hi, median, y)
    assert model.qhat[(0, -1)] == 10.0
    assert model.qhat[(0, 2)] == 1.0
    assert set(model.qhat) == {(0, -1), (0, 2)}
