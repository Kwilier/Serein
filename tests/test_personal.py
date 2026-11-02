import numpy as np

from serein.personal.action_plan import choose_zone, load_messages, make_plan
from serein.personal.model import ChildModel, design_row
from serein.personal.simulate import SimChild, simulate_log, simulate_pm_season


def _fit(or10, days, seed=0):
    beta = np.array([np.log(0.1 / 0.9), np.log(or10), 0.6 * np.log(or10),
                     0.3 * np.log(or10), np.log(3), np.log(1.5)])
    kid = SimChild(beta, 0.3, 0.9, "x")
    pm = simulate_pm_season(days, seed)
    log = simulate_log(kid, pm, seed)
    X = design_row(log["pm"], log["lag1"], log["lag2"], log["outdoor"], log["cold"], log["missed"])
    return ChildModel().fit(X, log["y"].to_numpy())


def test_model_separates_sensitive_from_insensitive_children():
    sens = np.mean([_fit(2.0, 300, s).sensitivity()["or_per_10"] for s in range(5)])
    insens = np.mean([_fit(1.0, 300, s).sensitivity()["or_per_10"] for s in range(5)])
    assert sens > 1.4 > insens


def test_uncertainty_shrinks_with_more_logged_days():
    a = _fit(1.3, 20).sensitivity()
    b = _fit(1.3, 300).sensitivity()
    assert (b["hi"] - b["lo"]) < (a["hi"] - a["lo"])


def test_zone_never_below_public_guidance():
    # Even a child with zero personal smoke risk gets the AQI floor.
    zone, reason = choose_zone(risk=0.05, baseline_risk=0.05, pm_median=45, pm_hi=60)
    assert zone == "yellow" and reason == "aqi_floor"
    zone, _ = choose_zone(risk=0.05, baseline_risk=0.05, pm_median=150, pm_hi=200)
    assert zone == "red"


def test_zone_escalates_on_personal_risk():
    zone, reason = choose_zone(risk=0.30, baseline_risk=0.10, pm_median=20, pm_hi=30)
    assert zone == "red" and reason == "personal"


def test_cold_alone_does_not_trigger_smoke_warning():
    m = ChildModel()
    plan = make_plan(m, "Sam", {0.05: 3, 0.5: 5, 0.95: 8}, 5, 5, cold=1)
    assert plan.zone == "green"


def test_every_language_has_every_key():
    en = load_messages("en")
    for lang in ("es",):
        other = load_messages(lang)
        assert set(other["zones"]) == set(en["zones"])
        assert set(other["categories"]) == set(en["categories"])
        for k in ("forecast_line", "risk_line", "emergency", "disclaimer"):
            assert k in other
        plan = make_plan(ChildModel(), "Ana", {0.05: 30, 0.5: 60, 0.95: 120}, 50, 50, language=lang)
        assert "911" in plan.message


def test_checkin_symptom_day_and_doctor_flag():
    from datetime import date

    import pytest

    from serein.personal.log import CheckIn

    quiet = CheckIn(date(2024, 9, 1), 0, 0, False, 1, False, False)
    assert quiet.symptom_day == 0 and not quiet.needs_doctor_flag
    puffs = CheckIn(date(2024, 9, 2), 0, 1, False, 1, False, False)
    assert puffs.symptom_day == 1
    severe = CheckIn(date(2024, 9, 3), 3, 0, False, 0, False, False)
    assert severe.needs_doctor_flag
    with pytest.raises(ValueError):
        CheckIn(date(2024, 9, 4), 5, 0, False, 0, False, False)


def test_log_joins_with_pm_and_fits():
    from datetime import date, timedelta

    from serein.personal.log import CheckIn, fit_child, model_rows, to_frame

    rng = np.random.default_rng(0)
    days = [date(2024, 7, 1) + timedelta(days=i) for i in range(90)]
    pm = {d: float(5 + 60 * (20 <= i < 30)) for i, d in enumerate(days)}  # one smoke episode
    entries = [CheckIn(d, int(pm[d] > 30 and rng.random() < 0.7), 0, False, 1, False, False)
               for d in days]
    log = to_frame(entries)
    _X, y = model_rows(log, pm)
    assert len(y) == 88  # first 2 days lack lag PM
    m = fit_child(log, pm)
    assert m.sensitivity()["or_per_10"] > 1.2
