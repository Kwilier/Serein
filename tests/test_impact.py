import math

import pandas as pd
import pytest

from serein.eval.impact import impact, summary_text


@pytest.mark.parametrize("field,value", [
    ("usg_events", -1), ("usg_events", 1.5), ("usg_events", float("inf")),
    ("n_area_days", -1), ("n_area_days", 20.5), ("n_area_days", float("nan")),
    ("usg_events", 21), ("usg_pod", -0.1), ("usg_pod", 1.1),
    ("usg_pod", float("nan")), ("usg_far", -0.1), ("usg_far", 1.1),
    ("usg_far", float("inf")), ("usg_far", float("nan")), ("usg_far", 1.0),
])
def test_impact_rejects_malformed_counts_and_ratios(tmp_path, field, value):
    rows = [
        {"who": who, "lead": 1, "usg_events": 10, "n_area_days": 20,
         "usg_pod": 0.5, "usg_far": 0.5}
        for who in ("smokecast", "epa_forecast")
    ]
    for row in rows:
        row[field] = value
    pd.DataFrame(rows).to_csv(tmp_path / "vs_epa_forecast.csv", index=False)

    with pytest.raises(ValueError):
        impact(tmp_path)


@pytest.mark.parametrize("events,pod,far", [(0, 0.0, 1.0), (10, 0.0, 0.5),
                                           (10, 0.0, 0.0), (10, 0.5, 0.9)])
def test_impact_rejects_impossible_warning_counts(tmp_path, events, pod, far):
    pd.DataFrame([
        {"who": who, "lead": 1, "usg_events": events, "n_area_days": 20,
         "usg_pod": pod, "usg_far": far}
        for who in ("smokecast", "epa_forecast")
    ]).to_csv(tmp_path / "vs_epa_forecast.csv", index=False)

    with pytest.raises(ValueError):
        impact(tmp_path)


@pytest.mark.parametrize("field", ["usg_events", "n_area_days"])
def test_impact_rejects_mismatched_populations(tmp_path, field):
    rows = [
        {"who": who, "lead": 1, "usg_events": 10, "n_area_days": 20,
         "usg_pod": 0.5, "usg_far": 0.5}
        for who in ("smokecast", "epa_forecast")
    ]
    rows[1][field] -= 1
    pd.DataFrame(rows).to_csv(tmp_path / "vs_epa_forecast.csv", index=False)

    with pytest.raises(ValueError, match="same.*population"):
        impact(tmp_path)


@pytest.mark.parametrize("models", [[], ["smokecast"], ["epa_forecast"],
                                   ["smokecast", "smokecast", "epa_forecast"]])
def test_impact_requires_one_row_per_model_and_lead(tmp_path, models):
    rows = [
        {"who": who, "lead": 1, "usg_events": 10, "n_area_days": 20,
         "usg_pod": 0.5, "usg_far": 0.5}
        for who in models
    ]
    rows.append({"who": "smokecast", "lead": 2, "usg_events": 10,
                 "n_area_days": 20, "usg_pod": 0.5, "usg_far": 0.5})
    pd.DataFrame(rows).to_csv(tmp_path / "vs_epa_forecast.csv", index=False)

    with pytest.raises(ValueError, match="exactly one.*lead 1"):
        impact(tmp_path)


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


@pytest.mark.parametrize("events", [0, 10])
def test_summary_explains_undefined_impact_values(tmp_path, events):
    pd.DataFrame([
        {"who": who, "lead": 2, "usg_events": events, "n_area_days": 20,
         "usg_pod": 0.0 if events else float("nan"), "usg_far": far}
        for who, far in [("smokecast", float("nan")), ("epa_forecast", 1.0)]
    ]).to_csv(tmp_path / "vs_epa_forecast.csv", index=False)

    text = summary_text(impact(tmp_path, lead=2))

    assert "nan" not in text.lower()
    assert "2 days ahead" in text
    assert "0 false alarms" in text
    assert "EPA's false-alarm count is unavailable" in text
    if events:
        assert "(0%)" in text
    else:
        assert "no smoky days to calculate warning percentages" in text


def test_summary_preserves_defined_comparisons():
    text = summary_text({
        "lead": 1, "n_area_days": 20, "smoky_area_days": 10,
        "serein": {"warned_days": 8, "false_alarms": 2},
        "epa": {"warned_days": 5, "false_alarms": 5},
        "pct_smoky_days_warned": (80, 50), "extra_warned_days": 3,
    })

    assert "a day ahead" in text
    assert "8 of them (80%) vs. 5 (50%)" in text
    assert "2 false alarms vs. 5" in text
