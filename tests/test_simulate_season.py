import numpy as np
import pytest

from serein.personal.simulate import simulate_pm_season


@pytest.mark.parametrize("n_days", [1, 2, 3, 4, 60, 150, np.int64(3)])
def test_simulated_seasons_support_short_and_full_histories(n_days):
    pm = simulate_pm_season(n_days, seed=3)

    assert pm.shape == (n_days,)
    assert np.isfinite(pm).all()
    assert (pm >= 0).all()
    np.testing.assert_array_equal(pm, simulate_pm_season(n_days, seed=3))


def test_existing_season_keeps_its_seeded_values():
    # Reference output from before short-history support was added.
    np.testing.assert_array_equal(
        simulate_pm_season(4, seed=0),
        [7.0, 162.8782129128045, 6.923259758421263, 8.152294429092821],
    )


@pytest.mark.parametrize("n_days", [0, -1, 1.0, 2.5, True, np.bool_(False), None,
                                      "3", np.nan, np.inf])
def test_simulated_seasons_reject_invalid_lengths(n_days):
    with pytest.raises(ValueError, match="n_days must be a positive integer"):
        simulate_pm_season(n_days)
