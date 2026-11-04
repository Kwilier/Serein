from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.features.daily import model_daily, obs_daily_means


@pytest.mark.parametrize("valid_hours", [17, 18])
def test_observation_completeness_counts_nonmissing_values(valid_hours):
    hourly = pd.DataFrame({
        "site_id": "A",
        "time_utc": pd.date_range("2023-08-01 08:00", periods=24, freq="h"),
        "pm25": [10.0] * valid_hours + [np.nan] * (24 - valid_hours),
    })
    row = obs_daily_means(hourly, pd.Series({"A": -8})).iloc[0]
    assert row["local_date"] == date(2023, 8, 1)
    assert row["n_hours"] == valid_hours
    if valid_hours < 18:
        assert np.isnan(row["pm25_daily"])
    else:
        assert row["pm25_daily"] == pytest.approx(10.0)


@pytest.mark.parametrize("valid_hours", [11, 12])
def test_forecast_completeness_counts_nonmissing_values_per_variable(valid_hours):
    times = pd.date_range("2023-08-01 08:00", periods=24, freq="h")
    forecast = pd.DataFrame({
        "issue_date": date(2023, 8, 1),
        "point_id": "A",
        "valid_utc": list(times) * 2,
        "var": ["smoke"] * 24 + ["t2m"] * 24,
        "value": [10.0] * valid_hours + [np.nan] * (24 - valid_hours) + [20.0] * 24,
    })
    out = model_daily(forecast, pd.Series({"A": -8}), group_extra=("var",))
    smoke = out.set_index("var").loc["smoke"]
    assert smoke["size"] == valid_hours
    if valid_hours < 12:
        assert np.isnan(smoke["mean"]) and np.isnan(smoke["max"])
    else:
        assert smoke["mean"] == pytest.approx(10.0)
        assert smoke["max"] == pytest.approx(10.0)
    temp = out.set_index("var").loc["t2m"]
    assert temp["size"] == 24
    assert temp["mean"] == pytest.approx(20.0)
