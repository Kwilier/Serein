from datetime import date

import numpy as np
import pandas as pd
import pytest

from serein.personal.log import CheckIn, model_rows, save_local, to_frame


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
