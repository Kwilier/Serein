from datetime import date, datetime

import pytest

from serein.data.availability import Availability


@pytest.mark.parametrize("field", ["issue_hour_utc", "obs_lag_hours", "firms_window_hours"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "12"])
def test_invalid_timing_settings(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        Availability(**{field: value})


@pytest.mark.parametrize("value", [24, 25])
def test_issue_hour_stays_on_issue_day(value):
    with pytest.raises(ValueError, match="issue_hour_utc"):
        Availability(issue_hour_utc=value)


@pytest.mark.parametrize("value", [-1, 0.5, float("nan"), True, "1"])
def test_hms_lag_is_nonnegative_whole_days(value):
    with pytest.raises((TypeError, ValueError), match="hms_lag_days"):
        Availability(hms_lag_days=value)


@pytest.mark.parametrize("field", ["hrrr_cycle", "aqm_cycle"])
def test_configured_cycles_are_validated(field):
    with pytest.raises(ValueError, match="cycle"):
        Availability(**{field: 24})


@pytest.mark.parametrize("lag", [-1, float("nan"), float("inf"), True, "2"])
def test_posting_lag_cannot_admit_future_runs(lag):
    with pytest.raises((TypeError, ValueError), match="posted_lag_hours"):
        Availability().check_model_cycle(date(2023, 9, 1), 12, lag)


def test_study_config_rejects_future_observations():
    with pytest.raises(ValueError, match="obs_lag_hours"):
        Availability.from_study({"availability": {"obs_lag_hours": -0.5}})


@pytest.mark.parametrize("cycle", [-1, 24, 12.5, True])
def test_model_posting_check_rejects_invalid_cycles(cycle):
    with pytest.raises((TypeError, ValueError), match="cycle"):
        Availability().check_model_cycle(date(2023, 9, 1), cycle, 2)


def test_fractional_and_zero_timing_boundaries():
    day = date(2023, 9, 1)
    av = Availability(issue_hour_utc=0.5, obs_lag_hours=0.5, hms_lag_days=0,
                      firms_window_hours=0)
    assert av.issue_time(day) == datetime(2023, 9, 1, 0, 30)
    assert av.last_obs_hour_start(day) == datetime(2023, 8, 31, 23)
    assert av.hms_day(day) == day
    assert av.firms_window(day) == (av.issue_time(day), av.issue_time(day))
    assert av.check_model_cycle(day, 0, 0.5)
    assert not av.check_model_cycle(day, 0, 0.51)
