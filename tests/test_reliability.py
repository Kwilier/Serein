from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from serein.eval.reliability import analyze_reliability, main, write_reliability


def predictions(probabilities, observations=None, *, start="2023-06-01", point="001"):
    if observations is None:
        observations = np.resize([10., 50.], len(probabilities))
    return pd.DataFrame({"issue_date": pd.date_range(start, periods=len(probabilities)), "point_id": point,
                         "y_L1": observations, "smokecast_prob_L1": probabilities})


def analyze(frame, **kwargs):
    return analyze_reliability(frame, n_boot=30, **kwargs)


def test_brier_and_bias_match_hand_calculation():
    frame = predictions([0., .25, .5, 1.], [10., 50., 10., 50.])
    result = analyze(frame, n_bins=2)
    summary = result["summary"].iloc[0]
    assert summary.n_scored == 4
    assert summary.n_events == 2
    assert summary.event_rate == .5
    assert summary.mean_probability == .4375
    assert summary.brier == .203125
    assert summary.probability_bias == -.0625
    assert summary.status == "ok"
    assert summary.bootstrap_status == "insufficient_blocks"
    assert np.isnan(summary.brier_ci_lo)
    bins = result["bins"].set_index("bin")
    assert bins.loc[0, "n_scored"] == bins.loc[1, "n_scored"] == 2
    assert bins.loc[0, "mean_probability"] == .125
    assert bins.loc[1, "mean_probability"] == .75
    assert bins.loc[0, "event_rate"] == bins.loc[1, "event_rate"] == .5
    assert bins.loc[-1, "n_input"] == 0
    assert summary.reliability == .1015625
    assert summary.resolution == 0
    assert summary.uncertainty == .25
    assert summary.brier_binned == .3515625
    assert summary.brier_grouping_difference == -.1484375


def test_perfect_and_opposite_probabilities():
    a = analyze(predictions([0., 1.]))["summary"].iloc[0]
    b = analyze(predictions([1., 0.]))["summary"].iloc[0]
    assert a.brier == 0
    assert b.brier == 1
    assert a.ece == a.reliability == 0
    assert b.ece == b.reliability == 1
    assert a.resolution == b.resolution == .25


@pytest.mark.parametrize("n_bins", [1, 2, 3, 10, 20])
def test_bin_boundaries_assign_to_right_and_include_one(n_bins):
    edges = np.arange(n_bins + 1) / n_bins
    result = analyze(predictions(edges), n_bins=n_bins)
    daily = result["daily"].sort_values("issue_date")
    assert daily.bin.tolist() == [*range(n_bins), n_bins - 1]
    assert result["bins"].query("bin >= 0").n_input.sum() == len(edges)


def test_boundary_neighbors_do_not_cross_wrong_edge():
    p = [np.nextafter(.3, 0), .3, np.nextafter(.3, 1)]
    daily = analyze(predictions(p))["daily"].sort_values("issue_date")
    assert daily.bin.tolist() == [2, 3, 3]


def test_observation_threshold_is_inclusive_and_not_a_warning_cutoff():
    frame = predictions([.1, .1, .1], [35.49, 35.5, 35.51])
    result = analyze(frame)["summary"].iloc[0]
    assert result.n_events == 2
    assert result.brier == pytest.approx((.01 + .81 + .81) / 3)
    other = analyze(frame, threshold=40)["summary"].iloc[0]
    assert other.n_events == 0
    assert other.brier == pytest.approx(.01)


@pytest.mark.parametrize("seed", range(6))
def test_binned_decomposition_matches_direct_bin_mean_forecasts(seed):
    rng = np.random.default_rng(seed)
    p, y = rng.random(103), rng.choice([0., 50.], 103)
    result = analyze(predictions(p, y), n_bins=7)
    summary, bins = result["summary"].iloc[0], result["bins"].set_index("bin")
    index = np.minimum(np.searchsorted(np.arange(8) / 7, p, side="right") - 1, 6)
    grouped = bins.mean_probability.loc[index].to_numpy()
    actual = np.mean((grouped - (y >= 35.5)) ** 2)
    assert summary.brier == pytest.approx(np.mean((p - (y >= 35.5)) ** 2))
    assert summary.brier_binned == pytest.approx(actual)
    assert summary.reliability - summary.resolution + summary.uncertainty == pytest.approx(actual)
    assert summary.brier - summary.brier_grouping_difference == pytest.approx(actual)


def test_common_population_and_missing_probability_audits():
    frame = predictions([.2, .3, np.nan, .4], [10., 50., 50., np.nan])
    frame["other_prob_L1"] = [.2, np.nan, .8, .4]
    pairwise = analyze(frame)["summary"].set_index("model")
    common = analyze(frame, common_rows=True)
    summary = common["summary"].set_index("model")
    assert pairwise.n_scored.to_dict() == {"other": 2, "smokecast": 2}
    assert summary.n_scored.eq(1).all()
    assert summary.n_input.eq(4).all()
    assert summary.n_observed.eq(3).all()
    assert summary.n_probability.eq(3).all()
    assert summary.n_pairwise.eq(2).all()
    assert summary.n_excluded_common.eq(1).all()
    assert summary.n_excluded.eq(3).all()
    assert summary.brier.tolist() == pytest.approx([.04, .04])
    bins = common["bins"].query("model == 'smokecast'")
    assert bins.n_input.sum() == 4
    assert bins.query("bin == -1").n_observed.iloc[0] == 1
    assert bins.query("bin == -1").n_scored.iloc[0] == 0
    assert bins.query("bin == 3").n_pairwise.iloc[0] == 1
    assert bins.query("bin == 3").n_scored.iloc[0] == 0


@pytest.mark.parametrize("missing", ["probabilities", "observations", "both"])
def test_no_scored_rows_remain_in_every_table(missing):
    frame = predictions([.2, .6])
    if missing in {"probabilities", "both"}:
        frame["smokecast_prob_L1"] = np.nan
    if missing in {"observations", "both"}:
        frame["y_L1"] = np.nan
    result = analyze(frame)
    summary = result["summary"].iloc[0]
    assert summary.status == "no_scored_rows"
    assert summary.n_scored == 0
    assert np.isnan(summary.brier)
    assert np.isnan(summary.ece)
    assert np.isnan(summary.brier_binned)
    assert result["bins"].n_input.sum() == 2
    assert result["daily"].n_input.sum() == 2


def test_groups_and_daily_sufficient_statistics_reproduce_summary():
    first = predictions([.1, .8] * 20)
    second = predictions([.3, .6] * 20, point="002")
    frame = pd.concat([first, second], ignore_index=True)
    frame["hms_prev"] = np.resize([0, 2, np.nan, 0], len(frame))
    result = analyze(frame, groups=["point", "month", "smoke"])
    expected_labels = {"all", "001", "002", "2023-06", "2023-07", "clear", "smoke", "unknown"}
    assert set(result["summary"].group_value) == expected_labels
    for row in result["summary"].itertuples():
        daily = result["daily"]
        daily = daily.loc[daily.group.eq(row.group) & daily.group_value.eq(row.group_value)]
        assert daily.n_scored.sum() == row.n_scored
        assert daily.n_events.sum() / row.n_scored == row.event_rate
        assert daily.brier_sum.sum() / row.n_scored == pytest.approx(row.brier)
        assert daily.probability_sum.sum() / row.n_scored == pytest.approx(row.mean_probability)


def test_bootstrap_matches_independent_calendar_block_oracle():
    dates = pd.to_datetime(["2023-01-01", "2023-01-10", "2023-02-20", "2023-03-15", "2023-04-18", "2023-05-22"])
    a = predictions([.1, .2, .3, .7, .8, .9])
    a["issue_date"] = dates
    b = a.copy()
    b["point_id"], b["smokecast_prob_L1"] = "002", .4
    frame = pd.concat([a, b], ignore_index=True)
    result = analyze_reliability(frame, n_bins=2, n_boot=40, seed=31, block_days=7)
    summary = result["summary"].iloc[0]
    days = result["daily"].groupby("issue_date").sum(numeric_only=True)
    assert summary.n_blocks == 6
    samples = []
    rng = np.random.default_rng(31)
    for _ in range(40):
        selected = rng.choice(len(days), size=len(days), replace=True)
        chunk = days.iloc[selected]
        samples.append([chunk.n_events.sum() / chunk.n_scored.sum(),
                        chunk.probability_sum.sum() / chunk.n_scored.sum(),
                        chunk.brier_sum.sum() / chunk.n_scored.sum()])
    bounds = np.percentile(samples, [2.5, 97.5], axis=0)
    for i, metric in enumerate(("event_rate", "mean_probability", "brier")):
        assert summary[f"{metric}_ci_lo"] == pytest.approx(bounds[0, i])
        assert summary[f"{metric}_ci_hi"] == pytest.approx(bounds[1, i])


def test_fewer_than_five_calendar_blocks_suppresses_bounds():
    frame = predictions(np.resize([.1, .8], 28))
    result = analyze(frame, block_days=365)["summary"].iloc[0]
    assert result.n_blocks == 1
    assert result.bootstrap_status == "insufficient_blocks"
    assert np.isnan(result.event_rate_ci_hi)


@pytest.mark.parametrize("observation", [0., 50.])
def test_constant_outcomes_do_not_produce_degenerate_confidence_bounds(observation):
    frame = predictions(np.resize([.1, .7], 42), [observation] * 42)
    result = analyze(frame)
    summary = result["summary"].iloc[0]
    assert summary.status == "ok"
    assert summary.n_scored == 42
    assert summary.n_blocks >= 5
    assert summary.bootstrap_status == "constant_outcome"
    assert summary.event_rate == int(observation >= 35.5)
    assert np.isfinite(summary.brier)
    for metric in ("event_rate", "mean_probability", "brier"):
        assert np.isnan(summary[f"{metric}_ci_lo"])
        assert np.isnan(summary[f"{metric}_ci_hi"])
    scored_bins = result["bins"].loc[lambda x: x.n_scored.gt(0)]
    assert scored_bins.bootstrap_status.eq("constant_outcome").all()
    assert scored_bins.event_rate_ci_lo.isna().all()
    assert scored_bins.event_rate_ci_hi.isna().all()


def test_row_order_missing_indices_and_input_preservation():
    frame = predictions(np.resize([.2, .7], 50))
    frame.index = [0] * len(frame)
    original = frame.copy(deep=True)
    expected = analyze(frame)
    actual = analyze(frame.sample(frac=1, random_state=3))
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("column", ["y_L1", "smokecast_prob_L1"])
@pytest.mark.parametrize("values", [["0.2", "0.4"], [0.2 + 0.1j, 0.4 + 0.2j]])
def test_text_and_complex_numbers_are_not_silently_converted(column, values):
    frame = predictions([.2, .4])
    frame[column] = values
    with pytest.raises(ValueError, match="real numeric"):
        analyze(frame)


def test_nullable_numeric_columns_and_boolean_probabilities():
    frame = predictions([0., 1., np.nan], [0., 50., np.nan])
    frame["y_L1"] = frame.y_L1.astype("Float64")
    frame["smokecast_prob_L1"] = pd.Series([False, True, pd.NA], dtype="boolean")
    result = analyze(frame)["summary"].iloc[0]
    assert result.n_scored == 2
    assert result.brier == 0


def test_multiple_leads_are_scored_separately():
    frame = predictions([.2, .7])
    frame["y_L2"], frame["smokecast_prob_L2"] = [50., 0.], [.9, .1]
    result = analyze(frame)["summary"].set_index("lead")
    assert result.loc[1, "brier"] == pytest.approx((.04 + .09) / 2)
    assert result.loc[2, "brier"] == pytest.approx(.01)
    assert result.loc[2, "n_scored"] == 2


@pytest.mark.parametrize("value", [-.01, 1.01, np.inf, -np.inf])
def test_invalid_probabilities_rejected_even_without_observations(value):
    frame = predictions([value], [np.nan])
    with pytest.raises(ValueError):
        analyze(frame)


@pytest.mark.parametrize("kwargs", [
    {"threshold": -1}, {"threshold": np.inf}, {"threshold": True},
    {"n_bins": 0}, {"n_bins": -1}, {"n_bins": 2.5}, {"n_bins": True},
    {"block_days": 0}, {"seed": -1}, {"common_rows": 1},
    {"groups": ["other"]}, {"groups": ["point", "point"]},
    {"models": []}, {"models": ["smokecast", "smokecast"]}, {"leads": [-1]},
])
def test_invalid_settings_fail(kwargs):
    with pytest.raises((ValueError, TypeError)):
        analyze(predictions([.3]), **kwargs)


@pytest.mark.parametrize("n_boot", [0, -1, 2.5, True])
def test_invalid_bootstrap_count(n_boot):
    with pytest.raises(ValueError):
        analyze_reliability(predictions([.3]), n_boot=n_boot)


@pytest.mark.parametrize("bad", ["empty", "duplicate", "reserved_issue", "reserved_target", "timezone",
                                "id", "no_probability", "no_target", "infinite_target", "smoke"])
def test_bad_inputs_fail_before_scoring(bad):
    frame = predictions([.1, .5])
    kwargs = {}
    if bad == "empty":
        frame = frame.iloc[:0]
    elif bad == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif bad == "reserved_issue":
        frame["issue_date"] = pd.date_range("2024-01-01", periods=2)
    elif bad == "reserved_target":
        frame["issue_date"] = pd.date_range("2023-12-30", periods=2)
    elif bad == "timezone":
        frame["issue_date"] = frame.issue_date.dt.tz_localize("UTC")
    elif bad == "id":
        frame["point_id"] = ["", "001"]
    elif bad == "no_probability":
        frame = frame.drop(columns="smokecast_prob_L1")
    elif bad == "no_target":
        frame = frame.drop(columns="y_L1")
    elif bad == "infinite_target":
        frame["y_L1"] = np.inf
    else:
        frame["hms_prev"] = [0, 9]
        kwargs["groups"] = ["smoke"]
    with pytest.raises(ValueError):
        analyze(frame, **kwargs)


def test_writer_records_settings_hash_and_does_not_overwrite(tmp_path):
    frame = predictions([.1, .8] * 20)
    source, output = tmp_path / "probabilities.parquet", tmp_path / "report"
    frame.to_parquet(source, index=False)
    payload = source.read_bytes()
    manifest = write_reliability(source, output, groups=["point"], n_boot=20)
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["models"] == ["smokecast"]
    assert manifest["leads"] == [1]
    assert manifest["threshold"] == 35.5
    assert manifest["groups"] == ["point"]
    assert manifest["n_boot"] == 20
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert source.read_bytes() == payload
    assert len(pd.read_csv(output / "summary.csv")) == 2
    assert (output / "daily.parquet").exists()
    with pytest.raises(FileExistsError):
        write_reliability(source, output)


def test_final_run_and_invalid_inputs_leave_output_absent(tmp_path):
    source, output = tmp_path / "missing.parquet", tmp_path / "report"
    (tmp_path / "FINAL_RUN_LOCK.json").write_text("{}")
    with pytest.raises(ValueError, match="development"):
        write_reliability(source, output)
    assert not output.exists()


def test_cli_returns_error_for_missing_probability_columns(tmp_path, capsys):
    frame = predictions([.2])
    frame = frame.drop(columns="smokecast_prob_L1")
    source, output = tmp_path / "predictions.parquet", tmp_path / "report"
    frame.to_parquet(source)
    with pytest.raises(SystemExit) as result:
        main(["--predictions", str(source), "--out", str(output)])
    assert result.value.code == 2
    assert "models must be nonempty" in capsys.readouterr().err
    assert not output.exists()
