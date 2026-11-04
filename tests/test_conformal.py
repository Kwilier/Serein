import numpy as np

from serein.models.conformal import CQR


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
