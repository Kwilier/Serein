from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.features.daily import model_daily, obs_daily_means


def test_subhourly_observations_do_not_satisfy_daily_completeness():
    hourly = pd.DataFrame({
        "site_id": "A",
        "time_utc": pd.date_range("2023-08-01 08:00", periods=18, freq="5min"),
        "pm25": 10.0,
    })
    original = hourly.copy(deep=True)
    row = obs_daily_means(hourly, pd.Series({"A": -8})).iloc[0]
    assert row["n_hours"] == 2
    assert np.isnan(row["pm25_daily"])
    pd.testing.assert_frame_equal(hourly, original)


def test_daily_observations_weight_hours_equally_with_subhourly_samples():
    hourly = pd.DataFrame({
        "site_id": "A",
        "time_utc": ["2023-08-01 08:00", "2023-08-01 08:05", "2023-08-01 09:00"],
        "pm25": [10.0, 30.0, 50.0],
    })
    row = obs_daily_means(hourly, pd.Series({"A": -8}), min_hours=2).iloc[0]
    assert row["n_hours"] == 2
    assert row["pm25_daily"] == pytest.approx(35.0)


@pytest.mark.parametrize("offset", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("kind", ["obs", "forecast"])
def test_daily_aggregation_rejects_nonfinite_site_offsets(kind, offset):
    offsets = pd.Series({"A": offset})
    frame = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01 08:00"],
                          "pm25": [10.0]})
    with pytest.raises(ValueError, match="offset"):
        if kind == "obs":
            obs_daily_means(frame, offsets, min_hours=1)
        else:
            forecast = frame.rename(columns={"site_id": "point_id", "time_utc": "valid_utc",
                                             "pm25": "value"})
            model_daily(forecast.assign(issue_date=date(2023, 8, 1)), offsets, min_count=1)


def test_daily_aggregation_ignores_unused_invalid_offsets():
    hourly = pd.DataFrame({"site_id": ["A", "outside"],
                           "time_utc": ["2023-08-01 08:00"] * 2, "pm25": [10.0, 100.0]})
    result = obs_daily_means(hourly, pd.Series({"A": -8.0, "unused": np.nan}), min_hours=1)
    assert result["point_id"].tolist() == ["A"]
    assert result["pm25_daily"].tolist() == [10.0]


@pytest.mark.parametrize("kind,hours", [("obs", 17), ("forecast", 11)])
def test_equivalent_timestamp_strings_count_as_one_hour(kind, hours):
    times = pd.date_range("2023-08-01 08:00", periods=hours, freq="h")
    strings = [t.isoformat() for t in times] + ["2023-08-01T01:00:00-07:00"]
    offsets = pd.Series({"A": -8})
    values = [10.0] * hours + [30.0]
    if kind == "obs":
        frame = pd.DataFrame({"site_id": "A", "time_utc": strings, "pm25": values})
        row = obs_daily_means(frame, offsets).iloc[0]
        assert row["n_hours"] == hours
        assert np.isnan(row["pm25_daily"])
        complete = obs_daily_means(frame, offsets, min_hours=1).iloc[0]
        assert complete["pm25_daily"] == pytest.approx((20 + 10 * (hours - 1)) / hours)
    else:
        frame = pd.DataFrame({"point_id": "A", "valid_utc": strings, "value": values,
                              "issue_date": date(2023, 8, 1)})
        row = model_daily(frame, offsets).iloc[0]
        assert row["size"] == hours
        assert np.isnan(row["mean"]) and np.isnan(row["max"])
        complete = model_daily(frame, offsets, min_count=1).iloc[0]
        assert complete["mean"] == pytest.approx((20 + 10 * (hours - 1)) / hours)
        assert complete["max"] == 20.0


@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles", "Asia/Kolkata"])
def test_daily_aggregation_uses_utc_instants_before_standard_time_offset(zone):
    times = pd.date_range("2023-08-01 08:00", periods=24, freq="h")
    aware = times.tz_localize("UTC").tz_convert(zone)
    offsets = pd.Series({"A": -8})
    hourly = pd.DataFrame({"site_id": "A", "time_utc": times, "pm25": np.arange(24.)})
    expected_obs = obs_daily_means(hourly, offsets)
    actual_obs = obs_daily_means(hourly.assign(time_utc=aware), offsets)
    pd.testing.assert_frame_equal(actual_obs, expected_obs)
    assert actual_obs["local_date"].tolist() == [date(2023, 8, 1)]
    assert actual_obs["n_hours"].tolist() == [24]

    forecast = hourly.rename(columns={"site_id": "point_id", "time_utc": "valid_utc",
                                       "pm25": "value"}).assign(issue_date=date(2023, 8, 1))
    expected_model = model_daily(forecast, offsets)
    actual_model = model_daily(forecast.assign(valid_utc=aware), offsets)
    pd.testing.assert_frame_equal(actual_model, expected_model)
    assert actual_model["lead"].tolist() == [0]
    assert actual_model["size"].tolist() == [24]


@pytest.mark.parametrize("hours", [11, 12])
def test_forecast_duplicates_do_not_inflate_daily_coverage(hours):
    forecast = pd.DataFrame({
        "issue_date": date(2023, 8, 1),
        "point_id": "A",
        "valid_utc": pd.date_range("2023-08-01 08:00", periods=hours, freq="h"),
        "value": 10.0,
    })
    forecast = pd.concat([forecast, forecast.iloc[[0]]], ignore_index=True)
    original = forecast.copy(deep=True)
    row = model_daily(forecast, pd.Series({"A": -8})).iloc[0]
    assert row["size"] == hours
    if hours < 12:
        assert np.isnan(row["mean"]) and np.isnan(row["max"])
    else:
        assert row["mean"] == pytest.approx(10.0)
        assert row["max"] == pytest.approx(10.0)
    pd.testing.assert_frame_equal(forecast, original)


def test_forecast_duplicate_averages_preserve_group_boundaries():
    forecast = pd.DataFrame({
        "issue_date": [date(2023, 8, 1)] * 5 + [date(2023, 7, 31)],
        "point_id": ["A"] * 4 + ["B", "A"],
        "var": ["smoke"] * 3 + ["t2m", "smoke", "smoke"],
        "valid_utc": pd.to_datetime([
            "2023-08-01 08:00", "2023-08-01 08:00", "2023-08-01 09:00",
            "2023-08-01 08:00", "2023-08-01 08:00", "2023-08-01 08:00",
        ]),
        "value": [10.0, 30.0, 40.0, 300.0, 100.0, 80.0],
    })
    out = model_daily(forecast, pd.Series({"A": -8, "B": -8}),
                      min_count=1, group_extra=("var",))
    rows = out.set_index(["issue_date", "point_id", "var", "lead"])
    smoke = rows.loc[(date(2023, 8, 1), "A", "smoke", 0)]
    assert smoke["size"] == 2
    assert smoke["mean"] == pytest.approx(30.0)
    assert smoke["max"] == pytest.approx(40.0)
    assert rows.loc[(date(2023, 8, 1), "A", "t2m", 0), "mean"] == 300.0
    assert rows.loc[(date(2023, 8, 1), "B", "smoke", 0), "mean"] == 100.0
    assert rows.loc[(date(2023, 7, 31), "A", "smoke", 1), "mean"] == 80.0


@pytest.mark.parametrize("hours", [17, 18])
def test_observation_duplicates_do_not_inflate_daily_coverage(hours):
    hourly = pd.DataFrame({
        "site_id": "A",
        "time_utc": pd.date_range("2023-08-01 08:00", periods=hours, freq="h"),
        "pm25": 10.0,
    })
    hourly = pd.concat([hourly, hourly.iloc[[0]]], ignore_index=True)
    original = hourly.copy(deep=True)
    row = obs_daily_means(hourly, pd.Series({"A": -8})).iloc[0]
    assert row["n_hours"] == hours
    if hours < 18:
        assert np.isnan(row["pm25_daily"])
    else:
        assert row["pm25_daily"] == pytest.approx(10.0)
    pd.testing.assert_frame_equal(hourly, original)


def test_observation_daily_mean_weights_unique_timestamps_equally():
    hourly = pd.DataFrame({
        "site_id": ["A", "A", "A", "A", "A", "B"],
        "time_utc": pd.to_datetime([
            "2023-08-01 08:00", "2023-08-01 08:00", "2023-08-01 09:00",
            "2023-08-01 09:00", "2023-08-02 08:00", "2023-08-01 08:00",
        ]),
        "pm25": [10.0, 30.0, 40.0, np.inf, 80.0, 100.0],
    })
    result = obs_daily_means(hourly, pd.Series({"A": -8, "B": -8}), min_hours=1)
    rows = result.set_index(["point_id", "local_date"])
    assert rows.loc[("A", date(2023, 8, 1)), "pm25_daily"] == pytest.approx(30.0)
    assert rows.loc[("A", date(2023, 8, 1)), "n_hours"] == 2
    assert rows.loc[("A", date(2023, 8, 2)), "pm25_daily"] == pytest.approx(80.0)
    assert rows.loc[("B", date(2023, 8, 1)), "pm25_daily"] == pytest.approx(100.0)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
@pytest.mark.parametrize("valid_hours", [17, 18])
def test_observation_completeness_excludes_infinite_readings(invalid, valid_hours):
    hourly = pd.DataFrame({
        "site_id": "A",
        "time_utc": pd.date_range("2023-08-01 08:00", periods=24, freq="h"),
        "pm25": [10.0] * valid_hours + [invalid] * (24 - valid_hours),
    })
    row = obs_daily_means(hourly, pd.Series({"A": -8})).iloc[0]
    assert row["n_hours"] == valid_hours
    if valid_hours < 18:
        assert np.isnan(row["pm25_daily"])
    else:
        assert row["pm25_daily"] == pytest.approx(10.0)
    assert np.isinf(hourly["pm25"]).sum() == 24 - valid_hours


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
@pytest.mark.parametrize("valid_hours", [11, 12])
def test_forecast_completeness_excludes_infinite_values(invalid, valid_hours):
    forecast = pd.DataFrame({
        "issue_date": date(2023, 8, 1),
        "point_id": "A",
        "valid_utc": pd.date_range("2023-08-01 08:00", periods=24, freq="h"),
        "value": [10.0] * valid_hours + [invalid] * (24 - valid_hours),
    })
    row = model_daily(forecast, pd.Series({"A": -8})).iloc[0]
    assert row["size"] == valid_hours
    if valid_hours < 12:
        assert np.isnan(row["mean"]) and np.isnan(row["max"])
    else:
        assert row["mean"] == pytest.approx(10.0)
        assert row["max"] == pytest.approx(10.0)
    assert np.isinf(forecast["value"]).sum() == 24 - valid_hours


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
