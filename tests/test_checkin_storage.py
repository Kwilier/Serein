from datetime import date

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
