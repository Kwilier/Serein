import numpy as np
import pytest

from serein.personal.action_plan import choose_zone, load_messages, make_plan
from serein.personal.model import ChildModel, design_row
from serein.personal.simulate import SimChild, simulate_log, simulate_pm_season


@pytest.mark.parametrize("with_entry", [False, True])
def test_empty_model_rows_preserve_population_prior(with_entry):
    from datetime import date

    from serein.personal.log import CheckIn, fit_child, model_rows, to_frame
    from serein.personal.model import FEATURES

    day = date(2023, 7, 1)
    entries = [CheckIn(day, 1, 0, False, 1, False, False)] if with_entry else []
    log = to_frame(entries)
    X, y = model_rows(log, {day: 20.0})
    assert X.shape == (0, len(FEATURES))
    assert y.shape == (0,)
    fitted = ChildModel().fit(X, y)
    from_log = fit_child(log, {day: 20.0})
    row = design_row(20, 10, 10, 0.3, 0, 0)
    prior_risk = ChildModel().risk(row)
    for model in (fitted, from_log):
        assert model.n_days == 0
        for key, value in model.risk(row).items():
            np.testing.assert_allclose(value, prior_risk[key])


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


@pytest.mark.parametrize("field,value", [
    ("symptoms", 0.5), ("symptoms", 2.5),
    ("puffs_bucket", 0.5), ("puffs_bucket", 2.5),
    ("symptoms", np.nan), ("puffs_bucket", np.inf),
    ("outdoor", 1.5),
])
def test_checkin_rejects_invalid_answer_choices(field, value):
    from datetime import date

    from serein.personal.log import CheckIn

    answers = {"symptoms": 0, "puffs_bucket": 0, "outdoor": 1}
    answers[field] = value
    with pytest.raises(ValueError, match="check-in answer out of range"):
        CheckIn(date(2023, 7, 1), night_wake=False, cold=False,
                missed_controller=False, **answers)


def test_log_keeps_last_submitted_checkin_for_each_day():
    from datetime import date, timedelta

    from serein.personal.log import CheckIn, to_frame

    entries = [
        CheckIn(date(2023, 7, 1) + timedelta(days=i % 3), i % 4, 0,
                False, 1, False, False)
        for i in range(40)
    ]
    expected = {entry.day: entry for entry in entries}
    log = to_frame(entries)
    assert list(log["day"]) == sorted(expected)
    assert list(log.index) == list(range(len(expected)))
    for row in log.itertuples():
        assert row.symptoms == expected[row.day].symptoms
        assert row.symptom_day == expected[row.day].symptom_day


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


def test_forecast_risk_counts_the_smoky_tail():
    # Same median, wider upper tail -> higher expected risk for a sensitive child.
    m = _fit(1.6, 120)
    narrow = m.risk_under_forecast({0.05: 30, 0.5: 40, 0.95: 50}, 20, 20, 0.3, 0, 0)
    wide = m.risk_under_forecast({0.05: 30, 0.5: 40, 0.95: 150}, 20, 20, 0.3, 0, 0)
    assert wide["mean"] > narrow["mean"]
    assert wide["hi"] > narrow["hi"]


def test_sms_is_short_and_has_stop_and_911():
    from serein.personal.action_plan import render_sms, sms_segments

    for lang in ("en", "es"):
        for zone, cat in (("green", 0), ("yellow", 2), ("red", 4)):
            t = render_sms(zone, "Alexandria", "day_after", 188, cat, lang,
                           link="serein.app/p/7Kq2")
            assert "STOP" in t
            assert ("911" in t) == (zone != "green")
            assert sms_segments(t)[2] <= 3, (lang, zone, sms_segments(t))


def test_sms_segment_counting():
    from serein.personal.action_plan import sms_segments

    assert sms_segments("a" * 160) == ("GSM-7", 160, 1)
    assert sms_segments("a" * 161) == ("GSM-7", 161, 2)
    assert sms_segments("ñ¿é" * 10)[0] == "GSM-7"
    assert sms_segments("día") == ("UCS-2", 3, 1)
    assert sms_segments("í" * 71) == ("UCS-2", 71, 2)
    assert sms_segments("€")[1] == 2   # extension table: 2 characters


@pytest.mark.parametrize("text,expected", [
    ("€" * 153, ("GSM-7", 306, 3)),
    ("a" * 152 + "€" + "a" * 152, ("GSM-7", 306, 3)),
    ("\U0001f600" * 67, ("UCS-2", 134, 3)),
    ("\U0001f600" * 66, ("UCS-2", 132, 2)),
])
def test_sms_parts_keep_mult_unit_characters_together(text, expected):
    from serein.personal.action_plan import sms_segments

    assert sms_segments(text) == expected


def test_confounders_raise_risk_but_never_trigger_a_smoke_warning():
    m = ChildModel()
    clean = {0.05: 3, 0.5: 5, 0.95: 8}
    plain = make_plan(m, "Sam", clean, 5, 5)
    # Ozone raises the predicted risk ...
    o3 = make_plan(m, "Sam", clean, 5, 5, ozone_ppb=80)
    assert o3.risk > plain.risk
    # ... but a clean-smoke day stays GREEN even on a hot, high-ozone, high-pollen day.
    worst = make_plan(m, "Sam", clean, 5, 5, ozone_ppb=90, hot_day=1, high_pollen=1)
    assert worst.zone == "green"
    # Unknown ozone (None) is the same as the old model with no ozone term.
    x_none = design_row(20, 10, 10, 0.3, 0, 0)
    x_low = design_row(20, 10, 10, 0.3, 0, 0, ozone_ppb=25)
    assert np.array_equal(x_none, x_low)
