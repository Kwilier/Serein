import math

import pandas as pd
import pytest

from serein.eval.impact import impact


@pytest.mark.parametrize("events", [0, 10])
def test_impact_handles_no_successful_warnings(tmp_path, events):
    pd.DataFrame([
        {"who": who, "lead": 1, "usg_events": events, "n_area_days": 20,
         "usg_pod": 0.0 if events else float("nan"), "usg_far": far}
        for who, far in [("smokecast", float("nan")), ("epa_forecast", 1.0)]
    ]).to_csv(tmp_path / "vs_epa_forecast.csv", index=False)

    result = impact(tmp_path)

    assert result["smoky_area_days"] == events
    for who in ("serein", "epa"):
        assert result[who]["warned_days"] == 0
        assert result[who]["missed_days"] == events
    assert result["serein"]["false_alarms"] == 0
    # A false-alarm ratio of one does not tell us how many warnings were issued.
    assert math.isnan(result["epa"]["false_alarms"])
    if events:
        assert result["pct_smoky_days_warned"] == (0, 0)
    else:
        assert all(math.isnan(value) for value in result["pct_smoky_days_warned"])


def test_impact_reconstructs_counts_for_smoky_days(tmp_path):
    pd.DataFrame([
        {"who": who, "lead": 1, "usg_events": 10, "n_area_days": 20,
         "usg_pod": pod, "usg_far": far}
        for who, pod, far in [("smokecast", 0.8, 0.2), ("epa_forecast", 0.5, 0.5)]
    ]).to_csv(tmp_path / "vs_epa_forecast.csv", index=False)

    result = impact(tmp_path)

    assert result["serein"] == pytest.approx(
        {"warned_days": 8, "missed_days": 2, "false_alarms": 2})
    assert result["epa"] == pytest.approx(
        {"warned_days": 5, "missed_days": 5, "false_alarms": 5})
    assert result["pct_smoky_days_warned"] == pytest.approx((80, 50))
