"""Calendar labels survive vectorized parsing without UTC date shifts."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from serein.features.daily import model_daily


@pytest.mark.parametrize("labels", [
    [date(2023, 8, 1), date(2023, 8, 2)],
    ["2023-08-01", "2023-08-02"],
    [datetime(2023, 8, 1, 23), datetime(2023, 8, 2, 1)],
    pd.date_range("2023-08-01", periods=2),
    pd.date_range("2023-08-01", periods=2, tz="America/Los_Angeles"),
    pd.Series(["2023-08-01", "2023-08-02"], dtype="category"),
    [np.datetime64("2023-08-01"), np.datetime64("2023-08-02")],
    [datetime(2023, 8, 1, 23, tzinfo=timezone(timedelta(hours=-7))),
     datetime(2023, 8, 2, 8, tzinfo=timezone(timedelta(hours=2)))],
    [pd.Timestamp("2023-08-01T23:00-07:00"), pd.Timestamp("2023-08-02T08:00+02:00")],
    [date(2023, 8, 1), "2023-08-02T01:00+14:00"],
])
def test_issue_date_labels_keep_their_own_day(labels):
    frame = pd.DataFrame({"issue_date": labels, "point_id": "A",
                          "valid_utc": "2023-08-02T12:00Z", "value": [10., 30.]})
    frame.index = [5, 2]
    original = frame.copy(deep=True)
    result = model_daily(frame, pd.Series({"A": -8}), min_count=1)
    assert result.issue_date.tolist() == [date(2023, 8, 1), date(2023, 8, 2)]
    assert result.lead.tolist() == [1, 0]
    assert result["mean"].tolist() == [10., 30.]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("missing", [None, pd.NaT, pd.NA, np.nan, "", "NaT", "nan"])
def test_vectorized_missing_labels_still_fail_before_grouping(missing):
    frame = pd.DataFrame({"issue_date": [date(2023, 8, 1), missing], "point_id": "A",
                          "valid_utc": ["2023-08-01T12:00Z", "2023-08-01T13:00Z"],
                          "value": [10., 30.]})
    with pytest.raises(ValueError, match="issue_date.*missing"):
        model_daily(frame, pd.Series({"A": -8}), min_count=1)


@pytest.mark.parametrize("invalid", ["not-a-date", "2023-02-30", True])
def test_invalid_labels_are_not_coerced_to_missing_or_other_days(invalid):
    frame = pd.DataFrame({"issue_date": [date(2023, 8, 1), invalid], "point_id": "A",
                          "valid_utc": "2023-08-01T12:00Z", "value": [10., 30.]})
    with pytest.raises((ValueError, TypeError)):
        model_daily(frame, pd.Series({"A": -8}), min_count=1)


def test_nonunique_input_index_does_not_realign_calendar_labels():
    frame = pd.DataFrame({"issue_date": [date(2023, 8, 1), date(2023, 8, 2)],
                          "point_id": "A", "valid_utc": "2023-08-02T12:00Z", "value": [10., 30.]},
                         index=[2, 2])
    result = model_daily(frame, pd.Series({"A": -8}), min_count=1)
    assert result.lead.tolist() == [1, 0]
    assert result["mean"].tolist() == [10., 30.]
