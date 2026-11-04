"""A reporting-area name must never combine predictions from different states."""

import warnings
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from serein.config import Region
from serein.eval import backtest

ISSUED = date(2023, 8, 1)
VALID = ISSUED + timedelta(days=1)
STUDY = SimpleNamespace(region=Region(lat_min=30, lat_max=50, lon_min=-125, lon_max=-110))


def _records(areas=(("Shared", "CA", 3), ("Shared", "OR", 0))):
    forecasts, observations = [], []
    for area, state, category in areas:
        forecasts.append({"as_of": ISSUED, "valid_date": VALID, "lead": 1,
                          "issue_date": ISSUED, "reporting_area": area, "state": state,
                          "lat": 38 if state.strip().upper() == "CA" else 45,
                          "lon": -121, "category_idx": category})
        observations.append({"valid_date": VALID, "reporting_area": area,
                             "state": state, "category_idx": category})
    return pd.DataFrame(forecasts), pd.DataFrame(observations)


def _compare(monkeypatch, points, records=None, warn=None):
    forecasts, observations = _records() if records is None else records
    monkeypatch.setattr(backtest, "load_forecasts", lambda days: forecasts)
    monkeypatch.setattr(backtest, "load_observed_area_aqi", lambda days: observations)
    test = pd.DataFrame({"issue_date": [ISSUED] * len(points), "point_id": points.index})
    pm = np.array([[100.] if point == "CA1" else [5.] for point in points.index])
    result = backtest.compare_with_epa(STUDY, test, pm, [1], points, warn)
    return result, forecasts, observations


def _points():
    return pd.DataFrame({"reporting_area": ["Shared", "Shared"], "state": ["CA", "OR"]},
                        index=pd.Index(["CA1", "OR1"], name="point_id"))


@pytest.mark.parametrize("with_warning_rule", [False, True])
def test_same_area_name_and_date_in_two_states_do_not_share_predictions(monkeypatch,
                                                                       with_warning_rule):
    warning_rule = np.array([[True], [False]]) if with_warning_rule else None
    result, _, _ = _compare(monkeypatch, _points(), warn=warning_rule)
    own = result.set_index("who").loc["smokecast"]

    assert own["n_area_days"] == 2
    assert own["cat_acc"] == 1
    assert own["usg_events"] == 1
    assert own["usg_pod"] == 1
    assert own["usg_far"] == 0


def test_area_and_state_keys_normalize_case_whitespace_and_duplicate_rows(monkeypatch):
    points = _points()
    points["reporting_area"] = ["  SHARED  ", "shared"]
    points["state"] = [" ca ", "or"]
    forecasts, observations = _records((("Shared", "CA", 3), (" SHARED ", "or ", 0)))
    forecasts = pd.concat([forecasts, forecasts.assign(reporting_area=" shared ")],
                            ignore_index=True)
    observations["reporting_area"] = "SHARED"
    original_fc, original_ob = forecasts.copy(deep=True), observations.copy(deep=True)
    result, _, _ = _compare(monkeypatch, points, (forecasts, observations))

    assert (result["n_area_days"] == 2).all()
    assert (result["cat_acc"] == 1).all()
    pd.testing.assert_frame_equal(forecasts, original_fc)
    pd.testing.assert_frame_equal(observations, original_ob)


def test_one_monitor_can_belong_to_multiple_reporting_areas_in_its_state(monkeypatch):
    points = _points()
    points.loc["CA1", "reporting_area"] = "Shared | Second | shared"
    records = _records((("Shared", "CA", 3), ("Shared", "OR", 0), ("Second", "CA", 3)))
    result, _, _ = _compare(monkeypatch, points, records)

    assert (result["n_area_days"] == 3).all()
    assert (result["cat_acc"] == 1).all()
    assert (result["usg_events"] == 2).all()


@pytest.mark.parametrize("state_column", [False, True])
def test_legacy_registry_can_infer_an_unambiguous_area_state(monkeypatch, state_column):
    points = _points().iloc[:1].copy()
    if state_column:
        points["state"] = np.nan
    else:
        points = points.drop(columns="state")
    with warnings.catch_warnings(record=True) as caught:
        result, _, _ = _compare(monkeypatch, points, _records((("Shared", "CA", 3),)))

    assert not caught
    assert (result["n_area_days"] == 1).all()
    assert (result["cat_acc"] == 1).all()


def test_legacy_registry_skips_ambiguous_area_states_with_rebuild_warning(monkeypatch):
    points = _points().drop(columns="state")
    with pytest.warns(UserWarning, match="Skipped 2.*serein build-points"):
        result, _, _ = _compare(monkeypatch, points)
    assert result.empty


def test_ambiguous_legacy_area_does_not_remove_other_unambiguous_areas(monkeypatch):
    points = _points().drop(columns="state")
    points.loc["CA1", "reporting_area"] = "Shared | Unique"
    records = _records((("Shared", "CA", 3), ("Shared", "OR", 0), ("Unique", "CA", 3)))
    with pytest.warns(UserWarning, match="unambiguous state"):
        result, _, _ = _compare(monkeypatch, points, records)
    assert (result["n_area_days"] == 1).all()
    assert (result["cat_acc"] == 1).all()
