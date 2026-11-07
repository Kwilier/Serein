from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from serein.eval.discrimination import (
    COUNTS,
    RATES,
    _bounds,
    analyze_discrimination,
    main,
    write_discrimination,
)


def predictions(p=(.1, .8, .4, .9), y=(10., 50., 50., 10.), *, start="2023-06-01"):
    return pd.DataFrame({"point_id": "001", "issue_date": pd.date_range(start, periods=len(p)),
                         "y_L1": y, "smokecast_prob_L1": p})


def analyze(frame, **kwargs):
    return analyze_discrimination(frame, n_boot=40, **kwargs)


def test_hand_calculated_rank_scores_and_inclusive_cutoff():
    result = analyze(predictions(), cutoffs=[.4, .9])
    summary = result["summary"].iloc[0]
    assert summary.roc_auc == .5
    assert summary.average_precision == pytest.approx(7 / 12)
    assert summary.n_events == summary.n_nonevents == 2
    curve = result["curves"]
    assert curve.warn_none.tolist() == [True, False, False, False, False]
    assert np.isnan(curve.cutoff.iloc[0])
    assert curve.cutoff.iloc[1:].tolist() == [.9, .8, .4, .1]
    assert curve.tp.tolist() == [0, 0, 1, 2, 2]
    assert curve.fp.tolist() == [0, 1, 1, 1, 2]
    fixed = result["thresholds"].set_index("cutoff")
    assert fixed.loc[.4, list(COUNTS)].tolist() == [2, 1, 0, 1]
    assert fixed.loc[.4, "recall"] == 1
    assert fixed.loc[.4, "precision"] == fixed.loc[.4, "csi"] == pytest.approx(2 / 3)
    assert fixed.loc[.4, "false_alarm_ratio"] == pytest.approx(1 / 3)
    assert fixed.loc[.4, "false_positive_rate"] == .5
    assert fixed.loc[.4, "warning_rate"] == .75
    assert fixed.loc[.9, list(COUNTS)].tolist() == [0, 1, 2, 1]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("tied", [False, True])
def test_ranking_matches_independent_library_oracles(seed, tied):
    rng = np.random.default_rng(seed)
    p = rng.random(107)
    if tied:
        p = np.round(p, 1)
    y = rng.choice([0., 50.], len(p))
    result = analyze(predictions(p, y))
    summary = result["summary"].iloc[0]
    assert summary.roc_auc == pytest.approx(roc_auc_score(y >= 35.5, p))
    assert summary.average_precision == pytest.approx(average_precision_score(y >= 35.5, p))
    curve = result["curves"]
    assert len(curve) == len(np.unique(p)) + 1
    for row in curve.itertuples(index=False):
        warning = np.zeros(len(p), bool) if row.warn_none else p >= row.cutoff
        assert row.tp == np.sum(warning & (y >= 35.5))
        assert row.fp == np.sum(warning & (y < 35.5))
        assert row.fn == np.sum(~warning & (y >= 35.5))
        assert row.tn == np.sum(~warning & (y < 35.5))


@pytest.mark.parametrize("p,auc,ap", [([0., 1., 0., 1.], 1., 1.),
                                       ([1., 0., 1., 0.], 0., .5),
                                       ([.5] * 4, .5, .5)])
def test_perfect_reversed_and_constant_ranking(p, auc, ap):
    summary = analyze(predictions(p, [0., 50., 0., 50.]))["summary"].iloc[0]
    assert summary.roc_auc == auc
    assert summary.average_precision == ap


def test_warn_none_endpoint_differs_from_probability_one_cutoff():
    result = analyze(predictions([1., 1.], [0., 50.]), cutoffs=[0., 1.])
    curve = result["curves"]
    assert curve.tp.tolist() == [0, 1]
    assert curve.fp.tolist() == [0, 1]
    assert np.isnan(curve.precision.iloc[0])
    assert result["thresholds"].tp.tolist() == [1, 1]


@pytest.mark.parametrize("kind,status", [("events", "no_nonevents"), ("nonevents", "no_events"),
                                         ("missing", "no_scored_rows")])
def test_missing_class_support_has_no_ranking_scores(kind, status):
    y = {"events": [50.] * 4, "nonevents": [0.] * 4, "missing": [np.nan] * 4}[kind]
    result = analyze(predictions(y=y))
    summary = result["summary"].iloc[0]
    assert summary.ranking_status == status
    assert np.isnan(summary.roc_auc) and np.isnan(summary.average_precision)
    assert result["thresholds"].n_input.eq(4).all()
    if kind == "missing":
        assert result["daily"].n_input.sum() == 4 * 5
        assert result["curves"][list(COUNTS)].sum().sum() == 0


def test_no_warnings_keeps_precision_and_false_alarm_ratio_undefined():
    row = analyze(predictions(), cutoffs=[1.])["thresholds"].iloc[0]
    assert row.warning_rate == row.recall == row.false_positive_rate == row.csi == 0
    assert np.isnan(row.precision) and np.isnan(row.false_alarm_ratio)


def test_population_audit_and_common_rows():
    frame = predictions([.1, np.nan, .4, .9], [10., 50., 50., np.nan])
    frame["other_prob_L1"] = [np.nan, .7, .8, .9]
    model_rows = analyze(frame)["summary"].set_index("model")
    common = analyze(frame, common_rows=True)
    summary = common["summary"].set_index("model")
    assert model_rows.n_scored.to_dict() == {"other": 2, "smokecast": 2}
    assert summary.n_scored.eq(1).all()
    assert summary.n_pairwise.eq(2).all()
    assert summary.n_input.eq(4).all()
    assert summary.n_observed.eq(3).all()
    assert summary.n_probability.eq(3).all()
    assert summary.n_excluded_common.eq(1).all()
    assert summary.n_excluded.eq(3).all()
    assert summary.ranking_status.eq("no_nonevents").all()
    daily = common["daily"].query("cutoff == 0.5").groupby("model").sum(numeric_only=True)
    assert daily.n_scored.eq(1).all()
    assert (daily.tp + daily.fp + daily.fn + daily.tn).eq(1).all()


def test_group_support_and_daily_audits_are_reconstructible():
    frame = predictions(np.linspace(0, 1, 42), [0., 50.] * 21, start="2023-05-15")
    frame["point_id"] = np.resize(["001", "002"], 42)
    frame["hms_prev"] = np.resize([0., 1., np.nan], 42)
    result = analyze(frame, groups=["point", "month", "smoke"], cutoffs=[.25, .75])
    assert set(result["summary"].group_value) >= {"001", "002", "2023-05", "2023-06", "unknown"}
    for row in result["thresholds"].itertuples(index=False):
        daily = result["daily"].query(
            "model == @row.model and lead == @row.lead and group == @row.group "
            "and group_value == @row.group_value and cutoff == @row.cutoff")
        assert daily[list(COUNTS)].sum().tolist() == [getattr(row, x) for x in COUNTS]
        assert daily.n_input.sum() == row.n_input
        assert daily.n_scored.sum() == row.n_scored
        assert row.tp + row.fn == row.n_events
        assert row.fp + row.tn == row.n_nonevents
        assert row.n_event_days <= row.n_days_scored


def test_input_order_index_and_selected_model_order_do_not_change_scores():
    frame = predictions(np.linspace(0, 1, 42), [0., 50.] * 21)
    frame["other_prob_L1"] = 1 - frame.smokecast_prob_L1
    expected = analyze(frame, models=["other", "smokecast"], block_days=2)
    shuffled = frame.sample(frac=1, random_state=6)
    shuffled.index = [0] * len(shuffled)
    actual = analyze(shuffled, models=["smokecast", "other"], block_days=2)
    for name in expected:
        keys = [x for x in ["model", "lead", "group", "group_value", "cutoff", "warn_none", "issue_date"]
                if x in expected[name]]
        pd.testing.assert_frame_equal(expected[name].sort_values(keys).reset_index(drop=True),
                                      actual[name].sort_values(keys).reset_index(drop=True))


def daily_counts(values, dates=None):
    result = pd.DataFrame(values, columns=COUNTS)
    result["issue_date"] = pd.date_range("2023-06-01", periods=len(values)) if dates is None else dates
    return result


def test_bootstrap_matches_independent_calendar_block_resampling():
    dates = pd.to_datetime(["2023-06-01", "2023-06-02", "2023-06-06", "2023-06-10", "2023-06-20", "2023-07-01"])
    daily = daily_counts([[1, 3, 2, 4], [3, 1, 1, 1], [2, 2, 1, 4], [1, 1, 3, 3],
                          [4, 3, 1, 1], [2, 1, 1, 1]], dates)
    actual = _bounds(daily, block_days=3, n_boot=51, seed=29)
    blocks = ((dates - pd.Timestamp("1970-01-01")).days // 3).to_numpy()
    sums = np.array([daily.loc[blocks == x, list(COUNTS)].sum() for x in sorted(set(blocks))])
    rng = np.random.default_rng(29)
    samples = []
    for _ in range(51):
        tp, fp, fn, tn = sums[rng.integers(len(sums), size=len(sums))].sum(0)
        samples.append([tp / (tp + fn), fp / (fp + tn), tp / (tp + fp), fp / (tp + fp),
                        (tp + fp) / (tp + fp + fn + tn), tp / (tp + fp + fn)])
    bounds = np.percentile(samples, [2.5, 97.5], axis=0)
    assert actual["n_blocks"] == len(sums)
    for index, name in enumerate(RATES):
        assert actual[f"{name}_ci_lo"] == pytest.approx(bounds[0, index])
        assert actual[f"{name}_ci_hi"] == pytest.approx(bounds[1, index])
        assert actual[f"{name}_n_boot_valid"] == 51


def test_rare_event_undefined_resamples_are_reported_without_conditional_bounds():
    daily = daily_counts([[0, 1, 0, 4]] * 9 + [[1, 1, 1, 2]])
    bounds = _bounds(daily, block_days=1, n_boot=500, seed=1)
    assert bounds["n_event_blocks"] == 1
    assert bounds["n_nonevent_blocks"] == 10
    assert bounds["bootstrap_status"] == "ok"
    assert bounds["recall_bootstrap_status"] == "undefined_resamples"
    assert 0 < bounds["recall_n_boot_valid"] < 500
    assert np.isnan(bounds["recall_ci_lo"])
    assert bounds["warning_rate_bootstrap_status"] == bounds["precision_bootstrap_status"] == "ok"


@pytest.mark.parametrize("counts", [[[0, 1, 0, 4]] * 8, [[1, 0, 2, 0]] * 8])
def test_constant_outcomes_do_not_claim_zero_width_bounds(counts):
    bounds = _bounds(daily_counts(counts), block_days=1, n_boot=40, seed=0)
    assert bounds["bootstrap_status"] == "constant_outcome"
    for name in RATES:
        assert np.isnan(bounds[f"{name}_ci_lo"])
        assert bounds[f"{name}_n_boot_valid"] == 0


@pytest.mark.parametrize("counts,boundary", [
    ([2, 0, 0, 3], ["recall", "false_positive_rate", "precision", "false_alarm_ratio", "csi"]),
    ([0, 2, 3, 4], ["recall", "precision", "false_alarm_ratio", "csi"]),
    ([2, 3, 0, 0], ["recall", "false_positive_rate", "warning_rate"]),
    ([0, 0, 2, 3], ["recall", "false_positive_rate", "warning_rate", "csi"]),
])
def test_boundary_rates_do_not_claim_zero_width_certainty(counts, boundary):
    bounds = _bounds(daily_counts([counts] * 8), block_days=1, n_boot=40, seed=0)
    assert bounds["bootstrap_status"] == "ok"
    for name in boundary:
        assert bounds[f"{name}_bootstrap_status"] == "boundary_rate"
        assert bounds[f"{name}_n_boot_valid"] == 40
        assert np.isnan(bounds[f"{name}_ci_lo"]) and np.isnan(bounds[f"{name}_ci_hi"])
    for name in set(RATES) - set(boundary):
        if name in {"precision", "false_alarm_ratio"} and counts[0] + counts[1] == 0:
            assert bounds[f"{name}_bootstrap_status"] == "undefined_resamples"
        else:
            assert bounds[f"{name}_bootstrap_status"] == "ok"
            assert np.isfinite(bounds[f"{name}_ci_lo"])


def test_empty_calendar_blocks_and_unscored_days_are_not_resampled():
    daily = daily_counts([[1, 1, 1, 1]] * 6 + [[0, 0, 0, 0]] * 4)
    a = _bounds(daily, block_days=1, n_boot=40, seed=0)
    b = _bounds(daily.iloc[:6], block_days=1, n_boot=40, seed=0)
    assert a == b
    assert a["n_blocks"] == 6


@pytest.mark.parametrize("kwargs", [
    {"threshold": -1}, {"threshold": np.nan}, {"threshold": True}, {"cutoffs": []},
    {"cutoffs": [True]}, {"cutoffs": [np.nan]}, {"cutoffs": [1.1]}, {"cutoffs": [-.1]},
    {"cutoffs": [1j]}, {"cutoffs": [".5"]}, {"cutoffs": [.1, .1]}, {"block_days": 0},
    {"block_days": 1.5}, {"block_days": True}, {"seed": -1}, {"common_rows": 1},
    {"groups": ["point", "point"]}, {"groups": ["year"]}, {"models": []},
    {"models": ["bad-name"]}, {"leads": []}, {"leads": [True]}, {"leads": [1, 1]},
])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        analyze(predictions(), **kwargs)


@pytest.mark.parametrize("n_boot", [0, -1, True, 1.5])
def test_invalid_bootstrap_count(n_boot):
    with pytest.raises(ValueError):
        analyze_discrimination(predictions(), n_boot=n_boot)


@pytest.mark.parametrize("column,value", [("smokecast_prob_L1", np.inf), ("smokecast_prob_L1", -.1),
                                          ("smokecast_prob_L1", 1.01), ("y_L1", -np.inf)])
def test_invalid_values_fail_even_without_matching_observation(column, value):
    frame = predictions(y=[np.nan] * 4)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        analyze(frame)


@pytest.mark.parametrize("column", ["smokecast_prob_L1", "y_L1"])
@pytest.mark.parametrize("kind", ["string", "complex"])
def test_nonreal_input_dtypes_fail(column, kind):
    frame = predictions()
    frame[column] = frame[column].astype(str if kind == "string" else complex)
    with pytest.raises(ValueError, match="real numeric"):
        analyze(frame)


@pytest.mark.parametrize("kind", ["duplicates", "missing_id", "blank_id", "numeric_id", "numeric_date",
                                  "time", "timezone", "missing_date", "duplicate_columns", "empty"])
def test_invalid_keys(kind):
    frame = predictions()
    if kind == "duplicates":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif kind == "missing_id":
        frame = frame.drop(columns="point_id")
    elif kind == "blank_id":
        frame["point_id"] = " "
    elif kind == "numeric_id":
        frame["point_id"] = 1
    elif kind == "numeric_date":
        frame["issue_date"] = np.arange(4)
    elif kind == "time":
        frame["issue_date"] += pd.Timedelta(hours=1)
    elif kind == "timezone":
        frame["issue_date"] = frame.issue_date.dt.tz_localize("UTC")
    elif kind == "missing_date":
        frame.loc[0, "issue_date"] = pd.NaT
    elif kind == "duplicate_columns":
        frame = pd.concat([frame, frame[["y_L1"]]], axis=1)
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        analyze(frame)


def test_development_guard_checks_issue_and_target_dates():
    with pytest.raises(ValueError, match="development issue"):
        analyze(predictions(start="2024-01-01"))
    with pytest.raises(ValueError, match="reserved target"):
        analyze(predictions(start="2023-12-28"))
    frame = predictions(start="2023-12-27")
    assert len(analyze(frame)["summary"]) == 1


def test_selected_leads_models_and_event_equality():
    frame = predictions(y=[35.49, 35.5, 35.51, 0.])
    frame["bad_prob_L2"] = -1.
    frame["y_L2"] = 100.
    result = analyze(frame, models=["smokecast"], leads=[1])
    assert result["summary"].n_events.tolist() == [2]
    assert analyze(frame, models=["smokecast"], leads=[1], threshold=40)["summary"].n_events.tolist() == [0]
    with pytest.raises(ValueError, match="missing probability"):
        analyze(frame)


def test_nullable_numeric_missing_inputs_and_nondefault_index():
    frame = predictions()
    frame.index = [0] * 4
    frame["y_L1"] = pd.array([0, 50, None, 50], dtype="Float64")
    frame["smokecast_prob_L1"] = pd.array([.1, None, .4, .9], dtype="Float64")
    summary = analyze(frame)["summary"].iloc[0]
    assert summary.n_scored == 2
    assert summary.roc_auc == summary.average_precision == 1.


def test_writer_round_trip_source_hash_and_no_overwrite(tmp_path):
    frame = predictions()
    source, output = tmp_path / "input.parquet", tmp_path / "report"
    frame.to_parquet(source)
    payload = source.read_bytes()
    expected = analyze(frame)
    manifest = write_discrimination(source, output, n_boot=40)
    assert source.read_bytes() == payload
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["models"] == ["smokecast"]
    assert manifest["leads"] == [1]
    assert manifest["cutoffs"] == [.1, .25, .5, .75, .9]
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name in ("curves", "daily"):
        pd.testing.assert_frame_equal(pd.read_parquet(output / f"{name}.parquet"), expected[name])
    for name in ("summary", "thresholds"):
        pd.testing.assert_frame_equal(pd.read_csv(output / f"{name}.csv"), expected[name], check_dtype=False)
    with pytest.raises(FileExistsError):
        write_discrimination(source, output)


def test_final_lock_and_invalid_source_create_no_output(tmp_path):
    source, output = tmp_path / "input.parquet", tmp_path / "report"
    predictions().to_parquet(source)
    (tmp_path / "FINAL_RUN_LOCK.json").write_text("{}")
    with pytest.raises(ValueError, match="development inputs"):
        write_discrimination(source, output)
    assert not output.exists()


def test_cli_explicit_cutoffs_and_clean_failure(tmp_path, capsys):
    source, output = tmp_path / "input.parquet", tmp_path / "report"
    predictions().to_parquet(source)
    assert main(["--predictions", str(source), "--out", str(output), "--model", "smokecast",
                 "--lead", "1", "--cutoff", "0.2", "--cutoff", "0.8", "--n-boot", "20",
                 "--block-days", "2", "--seed", "9", "--common-rows"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cutoffs"] == [.2, .8]
    assert manifest["common_rows"] is True
    assert manifest["block_days"] == 2
    assert manifest["seed"] == 9
    with pytest.raises(SystemExit) as exc:
        main(["--predictions", str(source), "--out", str(output)])
    assert exc.value.code == 2
    assert "output already exists" in capsys.readouterr().err
