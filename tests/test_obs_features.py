from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from serein.data.availability import Availability
from serein.features.table import _obs_features


@pytest.mark.parametrize("days", [
    ["2023-08-01T23:00:00-07:00", "2023-08-02T01:00:00+02:00"],
    [date(2023, 8, 1), pd.Timestamp("2023-08-02T01:00:00+02:00")],
])
def test_build_table_preserves_calendar_dates_across_timezones(monkeypatch, days):
    from serein.features.table import build_table

    requested = []

    def load_hourly(padded_days):
        requested.extend(padded_days)
        return pd.DataFrame()

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", load_hourly)
    original = days.copy()
    with pytest.raises(RuntimeError, match="No AirNow data"):
        build_table(SimpleNamespace(raw={}, leads=[0]), days,
                    pd.DataFrame({"point_id": ["A"], "gmt_offset": [-8]}))
    assert requested == [date(2023, 7, 29), date(2023, 7, 30), date(2023, 7, 31),
                         date(2023, 8, 1), date(2023, 8, 2), date(2023, 8, 3)]
    assert days == original


def test_build_table_rejects_duplicate_calendar_dates_across_timezones(monkeypatch):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("duplicate dates must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    with pytest.raises(ValueError, match="duplicate issue dates"):
        build_table(SimpleNamespace(raw={}, leads=[0]),
                    ["2023-08-01T01:00:00+02:00", "2023-08-01T23:00:00-07:00"],
                    pd.DataFrame({"point_id": ["A"], "gmt_offset": [-8]}))


@pytest.mark.parametrize("other", ["2023-08-01", pd.Timestamp("2023-08-01T12:00:00")])
def test_build_table_rejects_equivalent_issue_dates(monkeypatch, other):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("duplicate date requests must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    with pytest.raises(ValueError, match="duplicate issue dates"):
        build_table(SimpleNamespace(raw={}, leads=[0]), [date(2023, 8, 1), other],
                    pd.DataFrame({"point_id": ["A"], "gmt_offset": [-8]}))


@pytest.mark.parametrize("missing", [None, pd.NaT, ""])
def test_build_table_rejects_missing_issue_dates(monkeypatch, missing):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("missing date requests must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    with pytest.raises(ValueError, match="missing issue dates"):
        build_table(SimpleNamespace(raw={}, leads=[0]), [date(2023, 8, 1), missing],
                    pd.DataFrame({"point_id": ["A"], "gmt_offset": [-8]}))


def test_build_table_normalizes_issue_dates_without_mutating_request(monkeypatch):
    from serein.features.table import build_table

    days = ["2023-08-01", pd.Timestamp("2023-08-02T12:00:00")]
    original = days.copy()
    points = pd.DataFrame({"point_id": ["A"], "gmt_offset": [-8],
                           "lat": [40.0], "lon": [-120.0], "kind": ["epa"]})
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    monkeypatch.setattr("serein.features.table.airnow.load_hourly", lambda days: hourly)
    monkeypatch.setattr("serein.features.table.obs_daily_means", lambda *args: pd.DataFrame({
        "point_id": ["A", "A"], "local_date": [date(2023, 8, 1), date(2023, 8, 2)],
        "pm25_daily": [20.0, 30.0],
    }))
    monkeypatch.setattr("serein.features.table._model_features", lambda *args: pd.DataFrame())
    monkeypatch.setattr("serein.features.table.hms.density_at_points", lambda *args: [0])
    out = build_table(SimpleNamespace(raw={}, leads=[0], name="synthetic"), days, points,
                      use_firms=False, progress=False)
    assert out["issue_date"].tolist() == [date(2023, 8, 1), date(2023, 8, 2)]
    assert out["y_L0"].tolist() == [20.0, 30.0]
    assert days == original


@pytest.mark.parametrize("offset", [np.nan, np.inf, -np.inf])
def test_build_table_validates_all_offsets_before_loading_data(monkeypatch, offset):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid offsets must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    points = pd.DataFrame({"point_id": ["monitor", "grid"], "gmt_offset": [-8, offset]})
    original = points.copy(deep=True)
    with pytest.raises(ValueError, match="finite standard-time offset"):
        build_table(SimpleNamespace(raw={}, leads=[0, 1]), [date(2023, 8, 1)], points)
    pd.testing.assert_frame_equal(points, original)


@pytest.mark.parametrize("ids, message", [
    (["A", "A"], "duplicate point identifiers"),
    (["A", None], "missing point identifiers"),
])
def test_build_table_validates_point_ids_before_loading_data(monkeypatch, ids, message):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid point registries must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    points = pd.DataFrame({"point_id": ids, "gmt_offset": [-8, -7]})
    original = points.copy(deep=True)
    with pytest.raises(ValueError, match=message):
        build_table(SimpleNamespace(raw={}, leads=[0, 1]), [date(2023, 8, 1)], points)
    pd.testing.assert_frame_equal(points, original)


def test_build_table_rejects_empty_points_before_loading_data(monkeypatch):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("empty point requests must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    points = pd.DataFrame(columns=["point_id", "gmt_offset", "lat", "lon", "kind"])
    with pytest.raises(ValueError, match="at least one point"):
        build_table(SimpleNamespace(raw={}, leads=[0, 1]), [date(2023, 8, 1)], points)


def test_build_table_rejects_duplicate_days_before_loading_data(monkeypatch):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("duplicate date requests must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    day = date(2023, 8, 1)
    with pytest.raises(ValueError, match="duplicate issue dates"):
        build_table(SimpleNamespace(raw={}, leads=[0, 1]), [day, day],
                    pd.DataFrame({"point_id": ["A"], "gmt_offset": [-8]}))


def test_build_table_rejects_empty_days_before_loading_data(monkeypatch):
    from serein.features.table import build_table

    def unexpected_load(*args, **kwargs):
        pytest.fail("empty date requests must not load cached observations")

    monkeypatch.setattr("serein.features.table.airnow.load_hourly", unexpected_load)
    with pytest.raises(ValueError, match="at least one issue date"):
        build_table(SimpleNamespace(raw={}, leads=[0, 1]), [],
                    pd.DataFrame({"point_id": ["A"], "gmt_offset": [-8]}))


@pytest.mark.parametrize("missing", [None, pd.NaT, ""])
def test_daily_targets_reject_missing_dates(missing):
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    daily = pd.DataFrame({"point_id": ["A"], "local_date": [missing], "pm25_daily": [20.0]})
    points = pd.DataFrame({"point_id": ["A"]})
    with pytest.raises(ValueError, match="daily observations contain missing dates"):
        _obs_features(hourly, daily, points, [date(2023, 8, 1)], Availability(), [0])


def test_daily_targets_normalize_calendar_dates_without_mutating_input():
    day = date(2023, 8, 1)
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    daily = pd.DataFrame({
        "point_id": ["A", "A", "A"],
        "local_date": ["2023-07-31", pd.Timestamp("2023-08-01"), date(2023, 8, 2)],
        "pm25_daily": [10.0, 20.0, 30.0],
    })
    original = daily.copy(deep=True)
    points = pd.DataFrame({"point_id": ["A"]})
    row = _obs_features(hourly, daily, points, [day], Availability(), [0, 1]).iloc[0]
    assert row["obs_prevday"] == 10.0
    assert row["y_L0"] == 20.0
    assert row["y_L1"] == 30.0
    pd.testing.assert_frame_equal(daily, original)


def test_daily_targets_reject_duplicates_after_date_normalization():
    day = date(2023, 8, 1)
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    daily = pd.DataFrame({"point_id": ["A", "A"], "local_date": [day, "2023-08-01"],
                          "pm25_daily": [10.0, 30.0]})
    points = pd.DataFrame({"point_id": ["A"]})
    with pytest.raises(ValueError, match="duplicate.*daily"):
        _obs_features(hourly, daily, points, [day], Availability(), [0])


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


@pytest.mark.parametrize("values", [[10.0, 30.0], [10.0, 10.0], [10.0, np.nan]])
def test_rolling_observations_reject_duplicate_daily_targets(values):
    day = date(2023, 8, 1)
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    daily = pd.DataFrame({"point_id": ["A", "A"], "local_date": [day, day],
                          "pm25_daily": values})
    original = daily.copy(deep=True)
    points = pd.DataFrame({"point_id": ["A"]})
    with pytest.raises(ValueError, match="duplicate.*daily"):
        _obs_features(hourly, daily, points, [day], Availability(), [0])
    pd.testing.assert_frame_equal(daily, original)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_daily_targets_exclude_nonfinite_readings(invalid):
    day = date(2023, 8, 1)
    hourly = pd.DataFrame({"site_id": ["A"], "time_utc": ["2023-08-01T16:00:00"],
                           "pm25": [10.0]})
    daily = pd.DataFrame({
        "point_id": ["A", "A", "A", "A", "B"],
        "local_date": [date(2023, 7, 30), date(2023, 7, 31), day, date(2023, 8, 2), day],
        "pm25_daily": [invalid, invalid, invalid, 30.0, 20.0],
    })
    original = daily.copy(deep=True)
    points = pd.DataFrame({"point_id": ["A", "B"]})
    out = _obs_features(hourly, daily, points, [day], Availability(), [0, 1])
    row = out.set_index("point_id").loc["A"]
    for column in ("obs_prevday", "obs_prev2day", "y_L0"):
        assert np.isnan(row[column])
    assert row["y_L1"] == 30.0
    assert out.set_index("point_id").loc["B", "y_L0"] == 20.0
    pd.testing.assert_frame_equal(daily, original)
