from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.data.availability import Availability
from serein.features.table import _obs_features


@pytest.mark.parametrize("source", ["hourly", "daily", "points"])
def test_rolling_observations_reject_missing_point_identifiers(source):
    day = date(2023, 8, 1)
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    daily = pd.DataFrame({"point_id": ["A"], "local_date": [day], "pm25_daily": [20.0]})
    points = pd.DataFrame({"point_id": ["A"]})
    frame, column = {"hourly": (hourly, "site_id"), "daily": (daily, "point_id"),
                     "points": (points, "point_id")}[source]
    frame.loc[0, column] = None
    with pytest.raises(ValueError, match="missing point identifiers"):
        _obs_features(hourly, daily, points, [day], Availability(), [0])


def test_rolling_observations_reject_duplicate_output_points():
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    daily = pd.DataFrame(columns=["local_date", "point_id", "pm25_daily"])
    points = pd.DataFrame({"point_id": ["A", "A"]})
    with pytest.raises(ValueError, match="duplicate point identifiers"):
        _obs_features(hourly, daily, points, [date(2023, 8, 1)], Availability(), [0])


@pytest.mark.parametrize("missing", [None, pd.NaT, ""])
def test_rolling_observations_reject_missing_timestamps(missing):
    hourly = pd.DataFrame({
        "site_id": ["A", "A"],
        "time_utc": ["2023-08-01T16:00:00", missing],
        "pm25": [10.0, 100.0],
    })
    original = hourly.copy(deep=True)
    daily = pd.DataFrame(columns=["local_date", "point_id", "pm25_daily"])
    points = pd.DataFrame({"point_id": ["A"]})
    with pytest.raises(ValueError, match="missing.*timestamp"):
        _obs_features(hourly, daily, points, [date(2023, 8, 1)], Availability(), [0])
    pd.testing.assert_frame_equal(hourly, original)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_rolling_observations_exclude_nonfinite_readings(invalid):
    day = date(2023, 8, 1)
    av = Availability()
    cutoff = pd.Timestamp(av.last_obs_hour_start(day))
    hourly = pd.DataFrame({
        "site_id": "A",
        "time_utc": [cutoff - pd.Timedelta(hours=5),
                     cutoff - pd.Timedelta(hours=2), cutoff,
                     cutoff + pd.Timedelta(hours=1)],
        "pm25": [10.0, 20.0, invalid, 1000.0],
    })
    original = hourly.copy(deep=True)
    daily = pd.DataFrame(columns=["local_date", "point_id", "pm25_daily"])
    points = pd.DataFrame({"point_id": ["A", "B"]})
    out = _obs_features(hourly, daily, points, [day], av, [0]).set_index("point_id")
    row = out.loc["A"]
    assert np.isnan(row["obs_last1"])
    assert row["obs_last3"] == pytest.approx(20.0)
    assert row["obs_last6"] == pytest.approx(15.0)
    assert row["obs_last24"] == pytest.approx(15.0)
    assert row["obs_max24"] == pytest.approx(20.0)
    assert row["obs_n24"] == 2
    assert out.loc["B", "obs_n24"] == 0
    assert np.isnan(out.loc["B", "obs_last24"])
    pd.testing.assert_frame_equal(hourly, original)


@pytest.mark.parametrize("mixed", [False, True])
def test_rolling_observations_normalize_timestamps_before_grouping(mixed):
    day = date(2023, 8, 1)
    times = ["2023-08-01T08:00:00-07:00", "2023-08-01T09:00:00-07:00",
             "2023-08-01T09:30:00-07:00"]
    values = [10.0, 50.0, 1000.0]
    if mixed:
        times.insert(0, pd.Timestamp("2023-08-01T15:00:00"))
        values.insert(0, 30.0)
    hourly = pd.DataFrame({"site_id": "A", "time_utc": times, "pm25": values})
    original = hourly.copy(deep=True)
    daily = pd.DataFrame(columns=["local_date", "point_id", "pm25_daily"])
    points = pd.DataFrame({"point_id": ["A"]})
    row = _obs_features(hourly, daily, points, [day], Availability(), [0]).iloc[0]
    assert row["obs_n24"] == 2
    assert row["obs_last1"] == 50.0
    assert row["obs_last24"] == pytest.approx(35.0 if mixed else 30.0)
    assert row["obs_max24"] == 50.0
    pd.testing.assert_frame_equal(hourly, original)


def test_rolling_observations_weight_hours_equally_without_future_readings():
    day = date(2023, 8, 1)
    av = Availability()
    cutoff = pd.Timestamp(av.last_obs_hour_start(day))
    hourly = pd.DataFrame({
        "site_id": "A",
        "time_utc": [cutoff - pd.Timedelta(hours=2),
                     cutoff - pd.Timedelta(hours=2) + pd.Timedelta(minutes=30),
                     cutoff - pd.Timedelta(hours=1), cutoff,
                     cutoff + pd.Timedelta(minutes=30)],
        "pm25": [10.0, 30.0, 50.0, 80.0, 1000.0],
    })
    original = hourly.copy(deep=True)
    daily = pd.DataFrame(columns=["local_date", "point_id", "pm25_daily"])
    points = pd.DataFrame({"point_id": ["A"]})
    row = _obs_features(hourly, daily, points, [day], av, [0]).iloc[0]
    assert row["obs_n24"] == 3
    assert row["obs_last1"] == 80.0
    for column in ("obs_last3", "obs_last6", "obs_last24"):
        assert row[column] == pytest.approx(50.0)
    assert row["obs_max24"] == 80.0
    pd.testing.assert_frame_equal(hourly, original)
