import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from serein.eval.intervals import METRICS, analyze_intervals, main


def forecasts(n_days=42):
    days = pd.date_range("2023-06-01", periods=n_days)
    return pd.DataFrame({
        "issue_date": np.repeat(days, 2), "point_id": ["A", "B"] * n_days,
        "y_L1": np.tile([10., 30.], n_days),
        "smokecast_lo_L1": np.tile([10., 15.], n_days),
        "smokecast_hi_L1": np.tile([20., 25.], n_days),
        "hms_prev": np.tile([0., 2.], n_days),
    })


def test_coverage_tails_width_and_proper_interval_score():
    frame = forecasts(2)
    frame["y_L1"] = [10., 30., 5., 20.]
    summary, daily = analyze_intervals(frame, n_boot=20)
    row = summary.iloc[0]
    assert row.n_input == row.n_observed == row.n_bounds == row.n_scored == len(frame)
    assert row.coverage == pytest.approx(2 / len(frame))
    assert row.lower_miss_rate == row.upper_miss_rate == pytest.approx(1 / len(frame))
    assert row.mean_width == pytest.approx(np.mean([10., 10., 10., 10.]))
    expected = np.array([10., 10. + 2 / .1 * 5., 10. + 2 / .1 * 5., 10.])
    assert row.interval_score == pytest.approx(expected.mean())
    assert daily.score_sum.sum() == pytest.approx(expected.sum())
    assert row.bootstrap_status == "insufficient_blocks"
    assert all(pd.isna(row[f"{name}_ci_lo"]) for name in METRICS)


def test_missing_bounds_and_targets_keep_input_days_and_denominators():
    frame = forecasts(3)
    frame.loc[0, "smokecast_lo_L1"] = np.nan
    frame.loc[1, "smokecast_hi_L1"] = np.nan
    frame.loc[2, "y_L1"] = np.nan
    frame.loc[4:, "y_L1"] = np.nan
    summary, daily = analyze_intervals(frame)
    row = summary.iloc[0]
    assert row.n_input == len(frame)
    assert row.n_observed == frame.y_L1.notna().sum()
    assert row.n_bounds == frame[["smokecast_lo_L1", "smokecast_hi_L1"]].notna().all(axis=1).sum()
    assert row.n_scored == 1
    assert row.n_excluded == len(frame) - 1
    assert row.n_days_input == 3
    assert row.n_days_scored == 1
    assert daily.n_scored.tolist() == [0, 1, 0]
    assert daily.loc[daily.n_scored.eq(0), "score_sum"].eq(0).all()
    assert row.coverage == 0


def test_no_scoreable_rows_remain_explicit():
    frame = forecasts()
    frame["y_L1"] = np.nan
    summary, daily = analyze_intervals(frame, groups=["point"])
    assert summary.status.eq("no_scored_rows").all()
    assert summary.n_scored.eq(0).all()
    assert summary.n_blocks.eq(0).all()
    assert summary[list(METRICS)].isna().all().all()
    assert daily.n_input.sum() == len(frame) * 2


def test_groups_reveal_local_undercoverage_and_unknown_smoke():
    frame = forecasts()
    frame.loc[0, "hms_prev"] = np.nan
    summary, daily = analyze_intervals(frame, groups=["point", "month", "smoke"], n_boot=30)
    points = summary.loc[summary.group.eq("point")].set_index("group_value")
    assert points.loc["A", "coverage"] == 1
    assert points.loc["B", "coverage"] == 0
    smoke = summary.loc[summary.group.eq("smoke")].set_index("group_value")
    assert smoke.loc["unknown", "n_input"] == 1
    assert set(smoke.index) == {"smoke", "clear", "unknown"}
    assert smoke.n_input.sum() == len(frame)
    for group in ("all", "point", "month", "smoke"):
        assert summary.loc[summary.group.eq(group), "n_input"].sum() == len(frame)
        assert daily.loc[daily.group.eq(group), "n_input"].sum() == len(frame)


def test_common_rows_retains_pairwise_availability():
    frame = forecasts()
    frame["other_lo_L1"] = frame.smokecast_lo_L1
    frame["other_hi_L1"] = frame.smokecast_hi_L1
    frame.loc[0, "other_hi_L1"] = np.nan
    summary, _ = analyze_intervals(frame)
    pairwise = summary.set_index("model")
    assert pairwise.loc["smokecast", "n_scored"] == len(frame)
    assert pairwise.loc["other", "n_scored"] == len(frame) - 1
    summary, _ = analyze_intervals(frame, common_rows=True)
    common = summary.set_index("model")
    assert common.n_scored.eq(len(frame) - 1).all()
    assert common.loc["smokecast", "n_pairwise"] == len(frame)
    assert common.loc["smokecast", "n_excluded_common"] == 1
    assert common.loc["other", "n_excluded_common"] == 0
    assert common.coverage.nunique() == 1


def test_bootstrap_matches_independent_calendar_block_resampling():
    frame = forecasts(70)
    frame["y_L1"] += np.repeat(np.arange(70) % 3, 2) * 12
    frame.loc[::9, "y_L1"] = np.nan
    frame = frame.drop(index=np.arange(10, 30))
    n_boot, seed, block_days = 200, 37, 7
    summary, _ = analyze_intervals(frame, block_days=block_days, n_boot=n_boot, seed=seed)
    known = frame.dropna(subset=["y_L1"])
    blocks = (known.issue_date - pd.Timestamp("1970-01-01")).dt.days // block_days
    by_block = [rows for _, rows in known.groupby(blocks, sort=True)]
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_boot):
        sample = pd.concat([by_block[i] for i in rng.integers(0, len(by_block), len(by_block))])
        lo, hi, y = (sample[column].to_numpy() for column in
                     ("smokecast_lo_L1", "smokecast_hi_L1", "y_L1"))
        score = hi - lo + 2 / .1 * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        results.append([np.mean((lo <= y) & (y <= hi)), np.mean(y < lo), np.mean(y > hi),
                        np.mean(hi - lo), np.mean(score)])
    expected = np.percentile(results, [2.5, 97.5], axis=0)
    row = summary.iloc[0]
    assert row.n_blocks == len(by_block)
    assert row.bootstrap_status == "ok"
    for i, metric in enumerate(METRICS):
        assert row[f"{metric}_ci_lo"] == pytest.approx(expected[0, i])
        assert row[f"{metric}_ci_hi"] == pytest.approx(expected[1, i])


def test_bootstrap_clusters_sites_and_keeps_calendar_gaps():
    frame = forecasts(3)
    frame["issue_date"] = np.repeat(pd.to_datetime(["2023-06-01", "2023-08-01", "2023-10-01"]), 2)
    summary, _ = analyze_intervals(frame)
    assert summary.iloc[0].n_blocks == frame.issue_date.nunique()
    assert summary.iloc[0].bootstrap_status == "insufficient_blocks"
    repeated = pd.concat([frame.assign(point_id=frame.point_id + str(i)) for i in range(20)])
    other, _ = analyze_intervals(repeated)
    assert other.iloc[0].n_blocks == summary.iloc[0].n_blocks


def test_bootstrap_and_scores_are_invariant_to_row_order_and_input_is_unchanged():
    frame = forecasts(70)
    before = frame.copy(deep=True)
    summary, daily = analyze_intervals(frame, groups=["point", "smoke"], n_boot=50)
    shuffled, shuffled_daily = analyze_intervals(frame.sample(frac=1, random_state=9),
                                                groups=["point", "smoke"], n_boot=50)
    pd.testing.assert_frame_equal(summary, shuffled)
    pd.testing.assert_frame_equal(daily, shuffled_daily)
    pd.testing.assert_frame_equal(frame, before)


def test_selected_leads_and_models_do_not_require_point_forecasts():
    frame = forecasts()
    frame["y_L2"] = frame.y_L1
    frame["other_lo_L2"], frame["other_hi_L2"] = 0., 100.
    summary, _ = analyze_intervals(frame, models=["other"], leads=[2], alpha=.2)
    row = summary.iloc[0]
    assert row.model == "other" and row.lead == 2
    assert row.coverage == 1
    assert row.nominal_coverage == pytest.approx(1 - .2)


@pytest.mark.parametrize("change", ["duplicate keys", "duplicate columns", "empty id", "missing id",
                                    "timestamp", "timezone", "numeric date", "missing date", "empty"])
def test_invalid_prediction_keys_are_rejected(change):
    frame = forecasts()
    if change == "duplicate keys":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif change == "duplicate columns":
        frame = pd.concat([frame, frame[["y_L1"]]], axis=1)
    elif change == "empty id":
        frame.loc[0, "point_id"] = " "
    elif change == "missing id":
        frame.loc[0, "point_id"] = None
    elif change == "timestamp":
        frame["issue_date"] += pd.Timedelta(hours=1)
    elif change == "timezone":
        frame["issue_date"] = frame.issue_date.dt.tz_localize("UTC")
    elif change == "numeric date":
        frame["issue_date"] = 0
    elif change == "missing date":
        frame.loc[0, "issue_date"] = pd.NaT
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        analyze_intervals(frame)


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": 1}, {"alpha": np.nan}, {"alpha": True},
    {"block_days": 0}, {"block_days": 1.5}, {"block_days": True},
    {"n_boot": 0}, {"n_boot": False}, {"seed": -1}, {"seed": .5},
    {"common_rows": 1}, {"groups": ["point", "point"]}, {"groups": ["bogus"]},
    {"leads": []}, {"leads": [True]}, {"leads": [-1]}, {"leads": [1, 1]},
    {"models": []}, {"models": ["has space"]}, {"models": ["smokecast", "smokecast"]},
])
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        analyze_intervals(forecasts(), **kwargs)


@pytest.mark.parametrize("column", ["y_L1", "smokecast_lo_L1", "smokecast_hi_L1"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_are_not_silently_excluded(column, value):
    frame = forecasts()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        analyze_intervals(frame)


@pytest.mark.parametrize("missing_target", [False, True])
def test_reversed_intervals_are_rejected_even_without_a_target(missing_target):
    frame = forecasts()
    frame.loc[0, "smokecast_lo_L1"] = 100.
    if missing_target:
        frame.loc[0, "y_L1"] = np.nan
    with pytest.raises(ValueError, match="lower bounds"):
        analyze_intervals(frame)


@pytest.mark.parametrize("column", ["y_L1", "smokecast_lo_L1", "smokecast_hi_L1"])
def test_incomplete_exports_are_rejected(column):
    with pytest.raises(ValueError):
        analyze_intervals(forecasts().drop(columns=column), models=["smokecast"], leads=[1])


@pytest.mark.parametrize("hms", [-1, 4, .5, np.inf])
def test_unknown_hms_codes_are_rejected(hms):
    frame = forecasts()
    frame.loc[0, "hms_prev"] = hms
    with pytest.raises(ValueError):
        analyze_intervals(frame, groups=["smoke"])


def test_smoke_grouping_requires_hms_but_other_reports_do_not():
    frame = forecasts().drop(columns="hms_prev")
    assert len(analyze_intervals(frame)[0]) == 1
    with pytest.raises(ValueError, match="hms_prev"):
        analyze_intervals(frame, groups=["smoke"])


def test_overflowing_interval_score_is_rejected():
    frame = forecasts()
    frame.loc[0, "y_L1"] = np.finfo(float).max
    with pytest.raises(ValueError, match="overflow"):
        analyze_intervals(frame)


@pytest.fixture
def cli_files(tmp_path):
    source, config, output = (tmp_path / name for name in ("predictions.parquet", "study.yaml", "report"))
    forecasts().to_parquet(source, index=False)
    config.write_text(yaml.safe_dump({"split": {"calibrate": [2023], "test": [2024, 2025]}}))
    return source, config, output


def test_cli_writes_auditable_files_and_exact_source_hashes(cli_files):
    source, config, output = cli_files
    before = source.read_bytes()
    assert main([str(source.parent), "--config", str(config), "--output", str(output),
                 "--group", "point", "--group", "smoke", "--n-boot", "30"]) == 0
    summary = pd.read_csv(output / "summary.csv")
    daily = pd.read_parquet(output / "daily.parquet")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert manifest["models"] == ["smokecast"]
    assert manifest["leads"] == [1]
    assert source.read_bytes() == before
    assert set(summary.group) == {"all", "point", "smoke"}
    assert daily.n_input.sum() == manifest["n_input"] * len(set(summary.group))


@pytest.mark.parametrize("failure", ["existing output", "missing parent", "final lock", "test dates",
                                     "target in test", "missing config", "bad config", "bad yaml", "bad bounds"])
def test_cli_refuses_invalid_inputs_before_creating_output(cli_files, failure):
    source, config, output = cli_files
    if failure == "existing output":
        output.mkdir()
        (output / "sentinel").write_text("keep")
    elif failure == "missing parent":
        output = output / "nested"
    elif failure == "final lock":
        (source.parent / "FINAL_RUN_LOCK.json").write_text("{}")
    elif failure in {"test dates", "target in test"}:
        frame = forecasts()
        frame["issue_date"] = pd.Timestamp("2024-06-01" if failure == "test dates" else "2023-12-31")
        frame["point_id"] = [str(i) for i in range(len(frame))]
        frame.to_parquet(source, index=False)
    elif failure == "missing config":
        config = config.with_name("absent.yaml")
    elif failure == "bad config":
        config.write_text("split: {}")
    elif failure == "bad yaml":
        config.write_text("split: [unterminated")
    else:
        frame = forecasts()
        frame.loc[0, "smokecast_lo_L1"] = 100.
        frame.to_parquet(source, index=False)
    with pytest.raises(SystemExit) as error:
        main([str(source), "--config", str(config), "--output", str(output)])
    assert error.value.code == 2
    if failure == "existing output":
        assert (output / "sentinel").read_text() == "keep"
        assert len(list(output.iterdir())) == 1
    else:
        assert not output.exists()
