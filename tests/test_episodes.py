import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from serein.eval.episodes import analyze_episodes, main, plot_episode


def predictions(observed=(10, 40, 60, 40, 10), model=(40, 20, 45, 50, 5), *, lead=1):
    return pd.DataFrame({
        "point_id": ["A"] * len(observed),
        "issue_date": pd.date_range("2023-08-01", periods=len(observed)),
        f"y_L{lead}": np.asarray(observed, float), f"smokecast_L{lead}": np.asarray(model, float),
    })


def test_event_peak_and_onset_are_scored_on_valid_days():
    report = analyze_episodes(predictions())
    event = report.catalog.iloc[0]
    assert event.episode_id == "A:20230803"
    assert (event.start_date, event.end_date, event.n_days) == (
        pd.Timestamp("2023-08-03"), pd.Timestamp("2023-08-05"), 3)
    assert not event.left_censored and not event.right_censored
    score = report.scores.iloc[0]
    assert score.status == "complete"
    assert score.predicted_peak == 50
    assert score.peak_ratio == pytest.approx(50 / 60)
    assert score.peak_bias == -10
    assert score.observed_peak_ratio == .75
    assert score.peak_timing_error_days == 1
    assert score.onset_delay_days == 1
    assert score.warning_notice_days == 0
    assert score.first_warning_issue_date == pd.Timestamp("2023-08-03")
    assert not score.onset_warned
    summary = report.summary.iloc[0]
    assert (summary.daily_hits, summary.daily_misses, summary.daily_false_alarms,
            summary.daily_correct_negatives) == (2, 1, 1, 1)
    assert summary.daily_pod == pytest.approx(2 / 3)
    assert summary.daily_far == pytest.approx(1 / 3)
    assert summary.daily_csi == .5
    assert summary.onset_warning_rate == 0


def test_warned_onset_has_full_nominal_lead_notice():
    table = predictions(model=(10, 40, 45, 50, 5), lead=3)
    score = analyze_episodes(table).scores.iloc[0]
    assert score.onset_warned
    assert score.onset_delay_days == 0
    assert score.warning_notice_days == 3


def test_peak_ties_choose_earliest_date_after_sorting_and_do_not_mutate_input():
    table = predictions(observed=(10, 60, 60, 40, 10), model=(10, 50, 50, 40, 10))
    table = table.sample(frac=1, random_state=12)
    before = table.copy(deep=True)
    report = analyze_episodes(table)
    pd.testing.assert_frame_equal(table, before)
    score = report.scores.iloc[0]
    assert report.catalog.observed_peak_date.iloc[0] == pd.Timestamp("2023-08-03")
    assert score.predicted_peak_date == pd.Timestamp("2023-08-03")
    assert score.peak_timing_error_days == 0


@pytest.mark.parametrize("gap", ["absent", "missing"])
def test_unknown_day_splits_events_and_censors_both_boundaries(gap):
    table = predictions(observed=(10, 40, 60, 40, 10))
    if gap == "absent":
        table = table.drop(index=2)
    else:
        table.loc[2, "y_L1"] = np.nan
    catalog = analyze_episodes(table).catalog
    assert catalog.n_days.tolist() == [1, 1]
    assert catalog.left_censored.tolist() == [False, True]
    assert catalog.right_censored.tolist() == [True, False]


def test_clean_day_separates_two_uncensored_events():
    table = predictions(observed=(10, 40, 10, 40, 10))
    catalog = analyze_episodes(table).catalog
    assert catalog.n_days.tolist() == [1, 1]
    assert not catalog.left_censored.any()
    assert not catalog.right_censored.any()


def test_sites_are_independent_and_episode_ids_retain_monitor_ids():
    a = predictions()
    b = predictions(observed=(40, 40, 10, 10, 10), model=(10, 10, 10, 10, 10))
    b["point_id"] = "000100308"
    report = analyze_episodes(pd.concat([a, b], ignore_index=True))
    assert set(report.catalog.point_id) == {"A", "000100308"}
    assert report.summary.n_scored.iloc[0] == 2
    assert report.summary.detection_rate.iloc[0] == .5
    assert report.summary.n_censored.iloc[0] == 1


def test_overlapping_leads_define_one_catalog_and_prefer_known_observations():
    table = predictions(observed=(40, 60, 40, 10, 10))
    table["y_L2"] = [60, 40, 10, 10, 10]
    # The first copy of Aug 3 is missing, but lead 1 provides it.
    table.loc[0, "y_L2"] = np.nan
    table["smokecast_L2"] = [50, 50, 5, 5, 5]
    report = analyze_episodes(table, leads=[1, 2])
    assert len(report.catalog) == 1
    assert report.catalog.n_days.iloc[0] == 3
    assert report.scores.n_forecasts.tolist() == [3, 2]
    assert report.scores.status.tolist() == ["complete", "partial"]
    assert report.summary.n_scored.tolist() == [1, 0]
    assert len(report.daily.query("lead == 1")) == 6


def test_conflicting_targets_across_leads_are_rejected_even_when_not_scoring_that_lead():
    table = predictions()
    table["y_L2"] = [90, 40, 10, 10, 10]
    with pytest.raises(ValueError, match="conflicting observations"):
        analyze_episodes(table, leads=[1])


def test_partial_missing_and_complete_events_have_explicit_populations():
    table = predictions(observed=(10, 40, 60, 10, 50, 10, 70, 10),
                        model=(10, 40, np.nan, 10, np.nan, 10, 20, 10))
    table["noaa_best_L1"] = [10, 40, 50, 10, 40, 10, 80, 10]
    report = analyze_episodes(table)
    model = report.scores.query("model == 'smokecast'")
    assert model.status.tolist() == ["partial", "missing", "complete"]
    assert model.detected.iloc[0]
    assert pd.isna(model.detected.iloc[1])
    assert not model.detected.iloc[2]
    for row in report.summary.itertuples():
        assert row.n_scored == 1
        assert row.n_excluded == 2
    assert report.scores.scored.tolist() == [False, False, True, False, False, True]
    assert report.scores.exclusion_reason.tolist() == [
        "incomplete_forecast", "incomplete_forecast", "scored", "not_common", "not_common", "scored"]
    for (model_name, lead), rows in report.daily.groupby(["model", "lead"]):
        count = report.summary.loc[report.summary.model.eq(model_name) & report.summary.lead.eq(lead),
                                   "n_daily_scored"].iloc[0]
        assert rows.scored.sum() == count
    own = analyze_episodes(table, common_episodes=False)
    assert own.summary.n_scored.tolist() == [1, 3]
    assert report.summary.n_daily_scored.tolist() == [6, 6]
    assert own.summary.n_daily_scored.tolist() == [6, 8]
    assert report.summary.n_daily_excluded_common.tolist() == [0, 2]
    assert model.n_forecasts.sum() == 2


def test_uncensored_filter_preserves_catalog_and_audits():
    table = predictions(observed=(40, 40, 10, 40, 10))
    complete = analyze_episodes(table)
    uncensored = analyze_episodes(table, uncensored_only=True)
    pd.testing.assert_frame_equal(complete.catalog, uncensored.catalog)
    flags = ["scored", "exclusion_reason"]
    pd.testing.assert_frame_equal(complete.scores.drop(columns=flags), uncensored.scores.drop(columns=flags))
    assert complete.scores.scored.tolist() == [True, True]
    assert uncensored.scores.scored.tolist() == [False, True]
    assert uncensored.scores.exclusion_reason.tolist() == ["censored", "scored"]
    assert complete.summary.n_scored.iloc[0] == 2
    assert uncensored.summary.n_scored.iloc[0] == 1
    assert uncensored.summary.n_censored.iloc[0] == 1


def test_intervals_are_scored_on_individual_days_including_the_observed_peak():
    table = predictions()
    table["smokecast_lo_L1"] = [30, 10, 40, np.nan, 0]
    table["smokecast_hi_L1"] = [50, 30, 60, np.nan, 20]
    report = analyze_episodes(table)
    score = report.scores.iloc[0]
    assert score.n_intervals == 2
    assert score.interval_coverage == .5
    assert score.observed_peak_covered
    assert report.summary.n_peak_intervals.iloc[0] == 1
    assert report.summary.peak_interval_coverage.iloc[0] == 1
    assert report.summary.n_daily_intervals.iloc[0] == 4
    assert report.summary.daily_interval_coverage.iloc[0] == .5


def test_interval_can_exist_when_median_is_missing_without_scoring_episode():
    table = predictions(model=(40, 20, np.nan, 50, 5))
    table["smokecast_lo_L1"] = 0
    table["smokecast_hi_L1"] = 100
    report = analyze_episodes(table)
    assert report.scores.status.iloc[0] == "partial"
    assert report.scores.observed_peak_covered.iloc[0]
    assert report.summary.n_peak_intervals.iloc[0] == 0


def test_no_events_is_a_valid_report_with_false_alarm_scores():
    report = analyze_episodes(predictions(observed=(10, 10, 10, 10, 10)))
    assert report.catalog.empty and report.scores.empty
    assert report.summary.n_scored.iloc[0] == 0
    assert pd.isna(report.summary.detection_rate.iloc[0])
    assert report.summary.daily_false_alarms.iloc[0] == 3
    assert report.summary.daily_far.iloc[0] == 1


def test_no_forecasts_is_distinct_from_missed_episode():
    report = analyze_episodes(predictions(model=(np.nan,) * 5))
    assert report.scores.status.iloc[0] == "missing"
    assert report.summary.n_missing.iloc[0] == 1
    assert report.summary.n_detected.iloc[0] == 0
    assert pd.isna(report.summary.detection_rate.iloc[0])
    assert report.summary.n_daily_scored.iloc[0] == 0
    assert report.summary.n_daily_missing.iloc[0] == 5


def test_threshold_is_inclusive_and_configurable():
    table = predictions(observed=(10, 35.5, 10, 40, 10), model=(10, 35.5, 10, 40, 10))
    assert len(analyze_episodes(table).catalog) == 2
    assert len(analyze_episodes(table, threshold=40).catalog) == 1
    assert analyze_episodes(table).summary.detection_rate.iloc[0] == 1


@pytest.mark.parametrize("options", [
    {"threshold": 0}, {"threshold": -1}, {"threshold": np.nan}, {"threshold": np.inf},
    {"threshold": True}, {"models": []}, {"models": ["smokecast", "smokecast"]},
    {"models": ["../smokecast"]}, {"models": ["unknown"]},
    {"leads": []}, {"leads": [1, 1]}, {"leads": [True]}, {"leads": [1.5]},
    {"leads": [-1]}, {"leads": [2]}, {"common_episodes": "yes"}, {"uncensored_only": 1},
])
def test_invalid_options_are_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        analyze_episodes(predictions(), **options)


@pytest.mark.parametrize("column,value", [
    ("y_L1", np.inf), ("smokecast_L1", -np.inf),
    ("point_id", ""), ("point_id", None), ("point_id", 1),
    ("issue_date", pd.NaT), ("issue_date", "2023-08-01T03:00:00"),
])
def test_invalid_input_values_are_rejected(column, value):
    table = predictions()
    table[column] = table[column].astype(object)
    table.loc[0, column] = value
    with pytest.raises((TypeError, ValueError)):
        analyze_episodes(table)


def test_duplicate_keys_and_columns_are_rejected():
    table = predictions()
    with pytest.raises(ValueError, match="unique"):
        analyze_episodes(pd.concat([table, table.iloc[:1]]))
    with pytest.raises(ValueError, match="unique"):
        analyze_episodes(pd.concat([table, table[["y_L1"]]], axis=1))


@pytest.mark.parametrize("lo,hi", [(70, 80), (0, 30), (80, 70), (0, np.inf)])
def test_invalid_intervals_are_rejected(lo, hi):
    table = predictions()
    table["smokecast_lo_L1"], table["smokecast_hi_L1"] = 0., 100.
    table.loc[2, ["smokecast_lo_L1", "smokecast_hi_L1"]] = lo, hi
    with pytest.raises(ValueError):
        analyze_episodes(table)


def test_one_sided_intervals_are_rejected():
    table = predictions()
    table["smokecast_lo_L1"] = 0
    with pytest.raises(ValueError, match="both"):
        analyze_episodes(table)


def test_missing_and_empty_observations_have_clear_errors():
    with pytest.raises(ValueError, match="observation column"):
        analyze_episodes(predictions().drop(columns="y_L1"))
    for table in (predictions().iloc[:0], predictions(observed=(np.nan,) * 5)):
        with pytest.raises(ValueError, match="finite observation"):
            analyze_episodes(table)


@pytest.fixture
def cli_inputs(tmp_path):
    source = tmp_path / "predictions.parquet"
    predictions().to_parquet(source, index=False)
    config = tmp_path / "study.yaml"
    config.write_text(yaml.safe_dump({"split": {"calibrate": [2023], "test": [2024, 2025]}}))
    return source, config, tmp_path / "report"


def test_cli_writes_auditable_report_and_does_not_overwrite(cli_inputs):
    source, config, output = cli_inputs
    args = [str(source), "--config", str(config), "--output", str(output), "--lead", "1"]
    assert main(args) == 0
    assert {p.name for p in output.iterdir()} == {
        "catalog.csv", "scores.csv", "summary.csv", "daily.csv", "manifest.json"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["development_years"] == [2023]
    assert pd.read_csv(output / "summary.csv").n_scored.iloc[0] == 1
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize("reason", ["year", "lock", "invalid", "empty"])
def test_cli_refuses_non_development_or_invalid_inputs_before_writing(cli_inputs, reason):
    source, config, output = cli_inputs
    if reason == "lock":
        (source.parent / "FINAL_RUN_LOCK.json").write_text("{}")
    else:
        table = predictions()
        if reason == "year":
            table["issue_date"] = pd.date_range("2024-08-01", periods=5)
        elif reason == "invalid":
            table["smokecast_L1"] = table.smokecast_L1.astype(float)
            table.loc[0, "smokecast_L1"] = np.inf
        else:
            table = table.iloc[:0]
        table.to_parquet(source, index=False)
    with pytest.raises(SystemExit) as exc:
        main([str(source), "--config", str(config), "--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_issue_dates_accept_calendar_date_objects_and_reject_timezones():
    table = predictions()
    table["issue_date"] = [date(2023, 8, day) for day in range(1, 6)]
    assert len(analyze_episodes(table).catalog) == 1
    table["issue_date"] = pd.date_range("2023-08-01", periods=5, tz="UTC")
    with pytest.raises(ValueError, match="timezone-free"):
        analyze_episodes(table)


def test_signed_historical_values_are_retained_and_counted_without_clipping():
    table = predictions(observed=(-2, 40, 60, 40, 10), model=(-3, 20, 45, 50, 5))
    table["smokecast_lo_L1"], table["smokecast_hi_L1"] = -5., 100.
    report = analyze_episodes(table)
    assert report.daily.observed.iloc[0] == -2
    assert report.daily.predicted.iloc[0] == -3
    assert report.summary.n_daily_negative_observations.iloc[0] == 1
    assert report.summary.n_daily_negative_predictions.iloc[0] == 1


def test_close_targets_cannot_disagree_on_episode_threshold():
    table = predictions()
    table["y_L2"] = [40., 60., 40., 10., 10.]
    table.loc[1, "y_L1"] = 35.4999999
    table.loc[0, "y_L2"] = 35.5000001
    with pytest.raises(ValueError, match="threshold"):
        analyze_episodes(table)


def test_agreeing_rounded_targets_are_deterministic_after_row_shuffling():
    table = predictions()
    table["y_L2"] = [40.0000001, 60.0000001, 40., 10., 10.]
    expected = analyze_episodes(table, leads=[1])
    actual = analyze_episodes(table.sample(frac=1, random_state=7), leads=[1])
    pd.testing.assert_frame_equal(actual.catalog, expected.catalog)
    pd.testing.assert_frame_equal(actual.summary, expected.summary)


@pytest.mark.parametrize("seed", range(8))
def test_episode_catalog_matches_independent_calendar_scan(seed):
    rng = np.random.default_rng(seed)
    observed = rng.choice([5., 40., 80., np.nan], size=40)
    point = rng.choice([5., 40., 80., np.nan], size=40)
    table = predictions(observed=observed, model=point)
    table = table.drop(index=rng.choice(40, size=5, replace=False))
    report = analyze_episodes(table)
    truth = dict(zip(table.issue_date + pd.Timedelta(days=1), table.y_L1, strict=True))
    forecast = dict(zip(table.issue_date + pd.Timedelta(days=1), table.smokecast_L1, strict=True))
    expected = []
    active = []
    for day in pd.date_range(min(truth), max(truth) + pd.Timedelta(days=1)):
        value = truth.get(day, np.nan)
        if np.isfinite(value) and value >= 35.5:
            active.append(day)
        elif active:
            expected.append(active)
            active = []
    assert [(e.start_date, e.end_date, e.n_days) for e in report.catalog.itertuples()] == [
        (days[0], days[-1], len(days)) for days in expected]
    complete = [days for days in expected if all(np.isfinite(forecast[day]) for day in days)]
    summary = report.summary.iloc[0]
    assert summary.n_scored == len(complete)
    detected = sum(any(forecast[day] >= 35.5 for day in days) for days in complete)
    assert summary.n_detected == detected
    finite = np.isfinite(table.y_L1) & np.isfinite(table.smokecast_L1)
    assert summary.n_daily_scored == int(finite.sum())


def test_episode_plot_keeps_unknown_days_as_gaps_and_refuses_overwrite(tmp_path, monkeypatch):
    import matplotlib.figure
    import matplotlib.pyplot as plt

    table = predictions(observed=(10, 40, np.nan, 40, 10))
    report = analyze_episodes(table)
    output = tmp_path / "episode.png"
    original = matplotlib.figure.Figure.savefig
    plotted = []

    def inspect(figure, *args, **kwargs):
        observed = next(line for line in figure.axes[0].lines if line.get_label() == "Observed")
        plotted.append(dict(zip(pd.to_datetime(observed.get_xdata()), observed.get_ydata(), strict=True)))
        original(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", inspect)
    before = plt.get_fignums()
    plot_episode(report, "A:20230803", output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pd.isna(plotted[0][pd.Timestamp("2023-08-04")])
    assert plotted[0][pd.Timestamp("2023-08-03")] == 40
    assert plt.get_fignums() == before
    with pytest.raises(FileExistsError):
        plot_episode(report, "A:20230803", output)


def test_plotting_unknown_episode_does_not_create_a_file(tmp_path):
    output = tmp_path / "episode.png"
    with pytest.raises(ValueError, match="unknown episode"):
        plot_episode(analyze_episodes(predictions()), "not-an-event", output)
    assert not output.exists()


def test_cli_plots_are_indexed_safely_and_listed_in_manifest(cli_inputs):
    source, config, output = cli_inputs
    assert main([str(source), "--config", str(config), "--output", str(output),
                 "--plot-episode", "A:20230803"]) == 0
    assert (output / "episode_001.png").read_bytes().startswith(b"\x89PNG")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["plots"] == [{"episode_id": "A:20230803", "file": "episode_001.png"}]


@pytest.mark.parametrize("episode_ids", [["unknown"], ["A:20230803", "A:20230803"]])
def test_invalid_plot_requests_do_not_create_report(cli_inputs, episode_ids):
    source, config, output = cli_inputs
    args = [str(source), "--config", str(config), "--output", str(output)]
    for episode_id in episode_ids:
        args.extend(["--plot-episode", episode_id])
    with pytest.raises(SystemExit):
        main(args)
    assert not output.exists()


def test_development_issue_dates_cannot_score_targets_in_the_next_year(cli_inputs):
    source, config, output = cli_inputs
    table = predictions()
    table["issue_date"] = pd.date_range("2023-12-27", periods=5)
    table.to_parquet(source, index=False)
    with pytest.raises(SystemExit):
        main([str(source), "--config", str(config), "--output", str(output)])
    assert not output.exists()


def test_manifest_hash_identifies_decoded_bytes_when_source_is_replaced(cli_inputs, monkeypatch):
    from serein.eval import episodes

    source, config, output = cli_inputs
    original_bytes = source.read_bytes()
    original_analyze = episodes.analyze_episodes

    def analyze_then_replace(*args, **kwargs):
        report = original_analyze(*args, **kwargs)
        source.write_bytes(b"a different saved run appeared after loading")
        return report

    monkeypatch.setattr(episodes, "analyze_episodes", analyze_then_replace)
    assert main([str(source), "--config", str(config), "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert manifest["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
    assert pd.read_csv(output / "summary.csv").n_scored.iloc[0] == 1
