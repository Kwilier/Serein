from datetime import date

import numpy as np
import pandas as pd

from serein.personal.log import CheckIn, model_rows, save_local, to_frame


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
