import numpy as np
import pandas as pd
import pytest

from serein.personal.action_plan import choose_zone, load_messages, make_plan
from serein.personal.model import FEATURES, ChildModel, design_row
from serein.personal.simulate import SimChild, simulate_log, simulate_pm_season


@pytest.mark.parametrize("outcomes", [[0, 2], [-1, 1], [0.5, 1]])
def test_fit_rejects_nonbinary_outcomes_without_changing_model(outcomes):
    X = np.tile(design_row(20, 10, 10, 0.3, 0, 0), (2, 1))
    model = ChildModel().fit(X, np.array([0, 1]))
    mean, cov = model.mean.copy(), model.cov.copy()
    with pytest.raises(ValueError, match="outcomes must be 0 or 1"):
        model.fit(X, np.array(outcomes))
    np.testing.assert_array_equal(model.mean, mean)
    np.testing.assert_array_equal(model.cov, cov)
    assert model.n_days == 2


@pytest.mark.parametrize("X,y", [
    (np.zeros(len(FEATURES)), np.array([0])),
    (np.zeros((2, len(FEATURES) - 1)), np.array([0, 1])),
    (np.zeros((2, len(FEATURES))), np.array([[0], [1]])),
    (np.zeros((2, len(FEATURES))), np.array([0])),
])
def test_fit_rejects_malformed_training_shapes(X, y):
    with pytest.raises(ValueError, match="training shapes"):
        ChildModel().fit(X, y)


def test_fit_accepts_lists_and_filters_missing_rows():
    X = np.tile(design_row(20, 10, 10, 0.3, 0, 0), (4, 1))
    X[3, 1] = np.inf
    model = ChildModel().fit(X.tolist(), [False, True, np.nan, 1])
    expected = ChildModel().fit(X[:2], np.array([0, 1]))
    assert model.n_days == 2
    np.testing.assert_allclose(model.mean, expected.mean)
    np.testing.assert_allclose(model.cov, expected.cov)


@pytest.mark.parametrize("settings", [
    {"eval_days": 151}, {"eval_days": 150}, {"eval_days": 0},
    {"eval_days": -1}, {"eval_days": 1.5}, {"eval_days": True},
    {"n_days": 3}, {"n_days": np.inf}, {"n_children": 0},
    {"n_children": False}, {"ks": (-1,)}, {"ks": (1.5,)},
    {"ks": (True,)}, {"ks": ()},
])
def test_study_rejects_invalid_splits_before_simulation(monkeypatch, settings):
    from serein.personal import simulate

    def unexpected(*args, **kwargs):
        pytest.fail("invalid study settings must fail before simulating children")

    monkeypatch.setattr(simulate, "simulate_children", unexpected)
    with pytest.raises(ValueError):
        simulate.run_study(**settings)


def test_study_keeps_training_before_evaluation(monkeypatch):
    from serein.personal import simulate

    fitted_pm = []
    evaluated_pm = []
    original_fit = ChildModel.fit
    original_pred = simulate._pred
    original_log = simulate.simulate_log

    def fit(model, X, y):
        fitted_pm.append(X[:, 1].copy())
        return original_fit(model, X, y)

    def predict(model, X):
        evaluated_pm.append(X[:, 1].copy())
        return original_pred(model, X)

    monkeypatch.setattr(simulate, "simulate_pm_season",
                        lambda n_days, seed: np.arange(10, 10 + n_days, dtype=float))
    monkeypatch.setattr(simulate, "simulate_log",
                        lambda kid, pm, seed: original_log(kid, pm, seed, forecast_error_sd=0))
    monkeypatch.setattr(ChildModel, "fit", fit)
    monkeypatch.setattr(simulate, "_pred", predict)
    result = simulate.run_study(n_children=1, n_days=6, eval_days=2,
                                ks=(0, 2, 100), seed=0)
    assert result["k"].tolist() == [0, 2, 4, "oracle"]
    assert np.isfinite(result[["brier", "logloss"]].to_numpy()).all()
    assert len(fitted_pm) == 2
    np.testing.assert_allclose(fitted_pm[0], np.arange(10, 12) / 10)
    np.testing.assert_allclose(fitted_pm[1], np.arange(10, 14) / 10)
    assert len(evaluated_pm) == 3
    for pm in evaluated_pm:
        np.testing.assert_allclose(pm, np.arange(14, 16) / 10)


@pytest.mark.parametrize("pm", [[10.0], [10.0, 20.0], [10.0, 20.0, 40.0, 80.0]])
def test_simulated_logs_use_past_exposure_with_initial_value_padding(pm):
    kid = SimChild(np.zeros(6), 0.3, 0.9, "test")
    log = simulate_log(kid, np.array(pm), seed=0)
    assert len(log) == len(pm)
    np.testing.assert_array_equal(log["pm"], pm)
    np.testing.assert_array_equal(log["lag1"], [pm[max(0, i - 1)] for i in range(len(pm))])
    np.testing.assert_array_equal(log["lag2"], [pm[max(0, i - 2)] for i in range(len(pm))])


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


@pytest.mark.parametrize("field", ["night_wake", "cold", "missed_controller"])
@pytest.mark.parametrize("value", ["False", "True", None, 0, 1, np.nan])
def test_checkin_requires_boolean_yes_no_answers(field, value):
    from datetime import date

    from serein.personal.log import CheckIn

    answers = {"night_wake": False, "cold": False, "missed_controller": False}
    answers[field] = value
    with pytest.raises(ValueError, match="check-in yes/no answers must be boolean"):
        CheckIn(date(2023, 7, 1), 0, 0, outdoor=1, **answers)


def test_checkin_accepts_numpy_boolean_answers():
    from datetime import date

    from serein.personal.log import CheckIn

    entry = CheckIn(date(2023, 7, 1), 0, 0, np.bool_(False), 1,
                    np.bool_(True), np.bool_(False))
    assert entry.symptom_day == 0


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


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf, pd.NA])
@pytest.mark.parametrize("lag", [0, 1, 2])
def test_model_rows_skips_unavailable_exposure(missing, lag):
    from datetime import date, timedelta

    from serein.personal.log import CheckIn, model_rows, to_frame

    day = date(2023, 7, 3)
    log = to_frame([CheckIn(day, 1, 0, False, 1, False, False)])
    pm = {day - timedelta(days=i): 10.0 for i in range(3)}
    pm[day - timedelta(days=lag)] = missing
    X, y = model_rows(log, pm)
    assert X.shape == (0, len(FEATURES))
    assert y.shape == (0,)


def test_model_rows_uses_calendar_lags_and_keeps_valid_days():
    from datetime import date, timedelta

    from serein.personal.log import CheckIn, model_rows, to_frame

    start = date(2023, 7, 1)
    days = [start + timedelta(days=i) for i in range(7)]
    log = to_frame([CheckIn(d, i % 2, 0, False, 1, False, False)
                    for i, d in enumerate(days)])
    pm = {d: float(10 * i) for i, d in enumerate(days) if i != 3}
    X, y = model_rows(log, pm)
    expected = np.array([design_row(20, 10, 0, 0.3, 0, 0),
                         design_row(60, 50, 40, 0.3, 0, 0)])
    np.testing.assert_array_equal(X, expected)
    np.testing.assert_array_equal(y, [0, 0])


def test_forecast_risk_counts_the_smoky_tail():
    # Same median, wider upper tail -> higher expected risk for a sensitive child.
    m = _fit(1.6, 120)
    narrow = m.risk_under_forecast({0.05: 30, 0.5: 40, 0.95: 50}, 20, 20, 0.3, 0, 0)
    wide = m.risk_under_forecast({0.05: 30, 0.5: 40, 0.95: 150}, 20, 20, 0.3, 0, 0)
    assert wide["mean"] > narrow["mean"]
    assert wide["hi"] > narrow["hi"]


@pytest.mark.parametrize("name", [
    "pm_lag1", "pm_lag2", "outdoor_frac", "cold", "missed",
    "ozone_ppb", "hot_day", "high_pollen",
])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [0.0], "0"])
def test_forecast_risk_rejects_invalid_context(name, value):
    context = {"pm_lag1": 5, "pm_lag2": 5, "outdoor_frac": 0.3, "cold": 0, "missed": 0}
    context[name] = value
    with pytest.raises(ValueError, match="finite scalar"):
        ChildModel().risk_under_forecast({0.5: 10}, **context)


def test_forecast_risk_accepts_numpy_context_and_unknown_ozone():
    model = ChildModel()
    expected = model.risk_under_forecast({0.5: 10}, 5, 5, 0.3, 0, 0)
    actual = model.risk_under_forecast(
        {0.5: 10}, np.float64(5), np.int64(5), np.float64(0.3), False, False,
        ozone_ppb=None, hot_day=np.int64(0), high_pollen=False)
    assert actual == expected


@pytest.mark.parametrize("n_samples", [0, -1, 1.5, True, np.nan, np.inf, "10"])
@pytest.mark.parametrize("forecast", [False, True])
def test_risk_rejects_invalid_sample_counts(n_samples, forecast):
    model = ChildModel()
    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        if forecast:
            model.risk_under_forecast({0.5: 10}, 5, 5, 0.3, 0, 0,
                                      n_samples=n_samples)
        else:
            model.risk(design_row(10, 5, 5, 0.3, 0, 0), n_samples=n_samples)


@pytest.mark.parametrize("X", [
    np.zeros(len(FEATURES) - 1), np.zeros((1, 1, len(FEATURES))),
    np.full(len(FEATURES), np.nan), np.full(len(FEATURES), np.inf),
])
def test_risk_rejects_invalid_feature_rows(X):
    with pytest.raises(ValueError, match="risk features"):
        ChildModel().risk(X)


def test_risk_accepts_numpy_sample_count_and_empty_batch():
    model = ChildModel()
    row = design_row(10, 5, 5, 0.3, 0, 0)
    single = model.risk(row.tolist(), n_samples=np.int64(10))
    batch = model.risk(np.stack([row, row]), n_samples=10)
    for key in single:
        np.testing.assert_allclose(batch[key], np.repeat(single[key], 2))
        assert model.risk(np.empty((0, len(FEATURES))))[key].shape == (0,)


@pytest.mark.parametrize("quantiles", [
    {}, {0.0: 10}, {1.0: 10}, {np.nan: 10}, {np.inf: 10},
    {0.5: np.nan}, {0.5: np.inf}, {0.5: -1},
    {0.05: 30, 0.5: 20, 0.95: 40},
    {"0.5": 10}, {0.5: [10]}, {True: 10},
])
def test_forecast_risk_rejects_invalid_quantiles(quantiles):
    with pytest.raises(ValueError, match="forecast quantiles"):
        ChildModel().risk_under_forecast(quantiles, 5, 5, 0.3, 0, 0)


def test_forecast_risk_accepts_single_and_unordered_quantiles():
    model = ChildModel()
    single = model.risk_under_forecast({0.5: 10}, 5, 5, 0.3, 0, 0)
    constant = model.risk_under_forecast({0.95: 10, 0.05: 10, 0.5: 10},
                                         5, 5, 0.3, 0, 0)
    assert single == constant
    for value in single.values():
        assert np.isfinite(value)
    assert 0 <= single["lo"] <= single["mean"] <= single["hi"] <= 1


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
@pytest.mark.parametrize("field", ["risk", "baseline_risk", "pm_median", "pm_hi"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [0.1], "0.1", True])
def test_choose_zone_rejects_invalid_scalars(field, value):
    inputs = {"risk": 0.1, "baseline_risk": 0.1, "pm_median": 5.0, "pm_hi": 10.0}
    inputs[field] = value
    with pytest.raises(ValueError, match=field):
        choose_zone(**inputs)


@pytest.mark.parametrize("changes", [
    {"risk": -0.1}, {"risk": 1.1}, {"baseline_risk": -0.1},
    {"baseline_risk": 1.1}, {"pm_median": -1}, {"pm_hi": -1},
    {"pm_median": 20, "pm_hi": 10},
])
def test_choose_zone_rejects_invalid_ranges(changes):
    inputs = {"risk": 0.1, "baseline_risk": 0.1, "pm_median": 5.0, "pm_hi": 10.0}
    inputs.update(changes)
    with pytest.raises(ValueError):
        choose_zone(**inputs)


@pytest.mark.parametrize("risk,baseline,pm,expected", [
    (0, 0, 0, ("green", "personal_low")),
    (1, 1, 0, ("green", "personal_low")),
    (1, 0, 0, ("red", "personal")),
    (0, 0, 35.4, ("green", "personal_low")),
    (0, 0, 35.5, ("yellow", "aqi_floor")),
    (0, 0, 125.5, ("red", "aqi_floor")),
])
def test_choose_zone_accepts_endpoints_and_keeps_public_floors(risk, baseline, pm, expected):
    assert choose_zone(np.float64(risk), np.int64(baseline), pm, pm) == expected
