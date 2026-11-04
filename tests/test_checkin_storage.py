from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from serein.personal.log import CheckIn, model_rows, save_local, to_frame


@pytest.mark.parametrize("key", ["2026-11-04", datetime(2026, 11, 4),
                                 pd.Timestamp("2026-11-04"), pd.NaT, None,
                                 20261104, True, np.datetime64("2026-11-04")])
def test_model_rows_rejects_noncalendar_exposure_keys(key):
    log = to_frame([CheckIn(date(2026, 11, 4), 0, 0, False, 1, False, False)])
    before = log.copy(deep=True)
    pm = {key: 10.0}
    with pytest.raises(TypeError, match="PM history keys.*calendar date"):
        model_rows(log, pm)
    pd.testing.assert_frame_equal(log, before)
    assert list(pm) == [key]


@pytest.mark.parametrize("value", [True, np.bool_(False), -1.0, "10", 2 + 1j,
                                  [10.0], np.array([10.0]), np.array(10.0)])
@pytest.mark.parametrize("lag", [0, 1, 2])
def test_model_rows_rejects_invalid_exposure_values(value, lag):
    from datetime import timedelta

    day = date(2026, 11, 4)
    log = to_frame([CheckIn(day, 0, 0, False, 1, False, False)])
    before = log.copy(deep=True)
    pm = {day - timedelta(days=i): 10.0 for i in range(3)}
    pm[day - timedelta(days=lag)] = value
    with pytest.raises(ValueError, match="PM.*nonnegative.*number"):
        model_rows(log, pm)
    pd.testing.assert_frame_equal(log, before)


def test_model_rows_accepts_numpy_exposure_scalars_and_zero():
    day = date(2026, 11, 4)
    log = to_frame([CheckIn(day, 0, 0, False, 1, False, False)])
    pm = {date(2026, 11, 2): np.int64(0), date(2026, 11, 3): np.float32(10),
          day: np.float64(20)}
    X, y = model_rows(log, pm)
    expected_X, expected_y = model_rows(log, {d: float(v) for d, v in pm.items()})
    assert X.shape == (1, 9)
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)


@pytest.mark.parametrize("value", [
    None, np.nan, pd.NaT, 20261104, True, datetime(2026, 11, 4),
    pd.Timestamp("2026-11-04"), "2026-11-04T12:00:00", "2026-02-30", "",
])
def test_loaded_checkins_reject_invalid_dates_before_pm_lookup(value):
    log = to_frame([CheckIn(date(2026, 11, 4), 0, 0, False, 1, False, False)])
    log["day"] = pd.Series([value], dtype=object)
    before = log.copy(deep=True)
    error = ValueError if isinstance(value, str) else TypeError
    with pytest.raises(error, match="check-in day"):
        model_rows(log, {})
    pd.testing.assert_frame_equal(log, before)


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.0, -1, 3, "1", None, np.nan])
@pytest.mark.parametrize("has_pm", [False, True])
def test_loaded_checkins_reject_invalid_outdoor_categories(value, has_pm):
    log = to_frame([CheckIn(date(2026, 11, 4), 0, 0, False, 1, False, False)])
    log["outdoor"] = pd.Series([value], dtype=object)
    before = log.copy(deep=True)
    pm = {date(2026, 11, day): 10.0 for day in range(2, 5)} if has_pm else {}
    with pytest.raises(ValueError, match="outdoor"):
        model_rows(log, pm)
    pd.testing.assert_frame_equal(log, before)


@pytest.mark.parametrize("value", [True, np.bool_(False), 0.5, 1.0, -1, 2, "0", None, np.nan])
@pytest.mark.parametrize("has_pm", [False, True])
def test_loaded_checkins_reject_invalid_symptom_outcomes(value, has_pm):
    log = to_frame([CheckIn(date(2026, 11, 4), 0, 0, False, 1, False, False)])
    log["symptom_day"] = pd.Series([value], dtype=object)
    before = log.copy(deep=True)
    pm = {date(2026, 11, day): 10.0 for day in range(2, 5)} if has_pm else {}
    with pytest.raises(ValueError, match="symptom_day"):
        model_rows(log, pm)
    pd.testing.assert_frame_equal(log, before)


@pytest.mark.parametrize("outdoor", [0, 1, 2])
@pytest.mark.parametrize("outcome", [0, 1])
def test_loaded_checkins_accept_integer_categories(outdoor, outcome):
    log = to_frame([CheckIn(date(2026, 11, 4), outcome, 0, False, outdoor, False, False)])
    log["outdoor"] = pd.Series([np.int64(outdoor)], dtype=object)
    log["symptom_day"] = pd.Series([np.int64(outcome)], dtype=object)
    pm = {date(2026, 11, day): 10.0 for day in range(2, 5)}
    X, y = model_rows(log, pm)
    assert X.shape[0] == 1
    np.testing.assert_array_equal(y, [outcome])


@pytest.mark.parametrize("column", ["cold", "missed_controller"])
@pytest.mark.parametrize("value", [0.5, 2, "False", None, np.nan])
def test_loaded_checkins_reject_nonboolean_model_answers(column, value):
    log = to_frame([CheckIn(date(2026, 11, 4), 0, 0, False, 1, False, False)])
    log[column] = pd.Series([value], dtype=object)
    before = log.copy(deep=True)
    pm = {date(2026, 11, day): 10.0 for day in range(2, 5)}
    with pytest.raises(ValueError, match=column):
        model_rows(log, pm)
    pd.testing.assert_frame_equal(log, before)


def test_empty_checkins_preserve_columns_and_save_readable_csv(tmp_path):
    empty = to_frame([])
    populated = to_frame([CheckIn(date(2026, 11, 4), 0, 0, False, 1, False, False)])
    assert list(empty.columns) == list(populated.columns)
    path = tmp_path / "checkins.csv"
    save_local(empty, path)
    restored = pd.read_csv(path)
    assert restored.empty
    assert list(restored.columns) == list(populated.columns)
    X, y = model_rows(restored, {})
    assert X.shape[0] == y.shape[0] == 0


def test_saved_checkins_preserve_model_rows(tmp_path):
    log = to_frame([
        CheckIn(date(2026, 11, 4), 1, 0, False, 1, False, False),
        CheckIn(date(2026, 11, 5), 0, 0, False, 2, True, True),
    ])
    pm = {date(2026, 11, day): float(day * 2) for day in range(2, 6)}
    expected_X, expected_y = model_rows(log, pm)
    path = tmp_path / "checkins.csv"
    save_local(log, path)
    restored = pd.read_csv(path)
    before = restored.copy(deep=True)
    X, y = model_rows(restored, pm)
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)
    pd.testing.assert_frame_equal(restored, before)


def test_loaded_duplicate_checkins_use_latest_entry(tmp_path):
    first = CheckIn(date(2026, 11, 4), 0, 0, False, 0, False, False)
    replacement = CheckIn(date(2026, 11, 4), 2, 1, True, 2, True, True)
    next_day = CheckIn(date(2026, 11, 5), 0, 0, False, 1, False, False)
    expected = to_frame([first, next_day, replacement])
    loaded = pd.concat([to_frame([first, next_day]), to_frame([replacement])],
                       ignore_index=True)
    loaded.loc[2, "day"] = "2026-11-04"
    pm = {date(2026, 11, day): float(day * 2) for day in range(2, 6)}
    expected_X, expected_y = model_rows(expected, pm)
    path = tmp_path / "duplicates.csv"
    save_local(loaded, path)
    for frame in (loaded, pd.read_csv(path)):
        before = frame.copy(deep=True)
        X, y = model_rows(frame, pm)
        order = np.argsort(y)
        expected_order = np.argsort(expected_y)
        np.testing.assert_array_equal(X[order], expected_X[expected_order])
        np.testing.assert_array_equal(y[order], expected_y[expected_order])
        pd.testing.assert_frame_equal(frame, before)
