import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from serein.eval.episode_runs import combine_runs
from serein.eval.episodes import analyze_episodes, main


def candidate(values=(10., 20., 40., 50., 10.)):
    return pd.DataFrame({
        "point_id": ["A"] * 5, "issue_date": pd.date_range("2023-08-01", periods=5),
        "y_L1": [10., 40., 60., 40., 10.], "smokecast_L1": values,
        "smokecast_lo_L1": np.asarray(values) - 5,
        "smokecast_hi_L1": np.asarray(values) + 5,
    })


def test_candidates_use_common_episodes_and_keep_partial_forecasts_visible():
    reference = candidate()
    challenger = candidate((10., 50., 70., 50., 10.)).drop(index=2)
    before = challenger.copy(deep=True)
    table = combine_runs({"reference": reference, "challenger": challenger})
    pd.testing.assert_frame_equal(challenger, before)
    report = analyze_episodes(table, models=["reference", "challenger"])
    assert len(report.catalog) == 1
    assert report.scores.status.tolist() == ["complete", "partial"]
    assert report.summary.n_scored.tolist() == [0, 0]
    assert report.summary.n_daily_scored.tolist() == [4, 4]
    missing = report.daily.model.eq("challenger") & report.daily.valid_date.eq(pd.Timestamp("2023-08-04"))
    assert report.daily.loc[missing, "predicted"].isna().all()
    assert table.challenger_L1.isna().sum() == 1


def test_complete_candidates_have_paired_peak_and_interval_diagnostics():
    report = analyze_episodes(combine_runs({"w1": candidate(), "w6": candidate((10., 50., 70., 50., 10.))}),
                              models=["w1", "w6"])
    assert report.summary.n_scored.tolist() == [1, 1]
    assert report.scores.predicted_peak.tolist() == [50, 70]
    assert report.scores.prediction_at_observed_peak.tolist() == [40, 70]
    assert report.scores.n_intervals.tolist() == [3, 3]


def test_known_targets_fill_missing_copies_without_filling_forecasts():
    one, two = candidate(), candidate()
    two.loc[2, ["y_L1", "smokecast_L1"]] = np.nan
    table = combine_runs({"one": one, "two": two})
    assert table.loc[2, "y_L1"] == 60
    assert pd.isna(table.loc[2, "two_L1"])


def test_outer_alignment_retains_locations_and_dates_unique_to_each_run():
    one, two = candidate(), candidate()
    two["point_id"] = "B"
    table = combine_runs({"one": one, "two": two})
    assert len(table) == 10
    assert set(table.point_id) == {"A", "B"}
    assert table.one_L1.isna().sum() == table.two_L1.isna().sum() == 5


@pytest.mark.parametrize("change", [1., 1e-7])
def test_different_observed_targets_cannot_be_compared_as_the_same_study(change):
    two = candidate()
    two.loc[2, "y_L1"] += change
    with pytest.raises(ValueError, match="different observations"):
        combine_runs({"one": candidate(), "two": two})


@pytest.mark.parametrize("names", [["one"], ["y", "two"], ["has space", "two"], ["one", "one_lo"]])
def test_invalid_or_colliding_candidate_names_are_rejected(names):
    with pytest.raises(ValueError):
        combine_runs({name: candidate() for name in names})


@pytest.mark.parametrize("change", ["missing model", "missing observations", "duplicate keys", "infinity"])
def test_invalid_candidate_tables_are_rejected(change):
    two = candidate()
    if change == "missing model":
        two = two.drop(columns="smokecast_L1")
    elif change == "missing observations":
        two = two.drop(columns="y_L1")
    elif change == "duplicate keys":
        two = pd.concat([two, two.iloc[:1]])
    else:
        two.loc[0, "smokecast_L1"] = np.inf
    with pytest.raises(ValueError):
        combine_runs({"one": candidate(), "two": two})


@pytest.fixture
def cli_sources(tmp_path):
    one, two, config = (tmp_path / name for name in ("one.parquet", "two.parquet", "study.yaml"))
    candidate().to_parquet(one, index=False)
    candidate((10., 50., 70., 50., 10.)).to_parquet(two, index=False)
    config.write_text(yaml.safe_dump({"split": {"calibrate": [2023]}}))
    return one, two, config, tmp_path / "report"


def test_cli_combines_sources_and_records_all_hashes(cli_sources):
    one, two, config, output = cli_sources
    assert main([str(one), "--candidate", f"w6={two}", "--reference-name", "w1",
                 "--config", str(config), "--output", str(output)]) == 0
    summary = pd.read_csv(output / "summary.csv")
    assert summary.model.tolist() == ["w1", "w6"]
    assert summary.n_scored.tolist() == [1, 1]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["candidate_source_model"] == "smokecast"
    assert [(source["name"], source["sha256"]) for source in manifest["sources"]] == [
        ("w1", hashlib.sha256(one.read_bytes()).hexdigest()),
        ("w6", hashlib.sha256(two.read_bytes()).hexdigest())]


@pytest.mark.parametrize("options", [
    ["--candidate", "missing-equals"], ["--candidate", "=path"],
    ["--candidate", "reference=ignored"], ["--candidate", "w6=ignored", "--model", "a", "--model", "b"],
])
def test_cli_rejects_invalid_candidate_requests_before_writing(cli_sources, options):
    one, _, config, output = cli_sources
    with pytest.raises(SystemExit):
        main([str(one), "--config", str(config), "--output", str(output), *options])
    assert not output.exists()


def test_candidate_cannot_supply_reserved_issue_dates(cli_sources):
    one, two, config, output = cli_sources
    table = candidate()
    table["issue_date"] = pd.date_range("2024-08-01", periods=5)
    table.to_parquet(two, index=False)
    with pytest.raises(SystemExit):
        main([str(one), "--candidate", f"new={two}", "--config", str(config), "--output", str(output)])
    assert not output.exists()


def test_extra_observation_leads_do_not_add_unselected_forecast_leads(cli_sources):
    one, two, config, output = cli_sources
    table = candidate()
    table["y_L2"] = [40., 60., 40., 10., 10.]
    table["smokecast_L2"] = [50., 50., 40., 10., 10.]
    table.to_parquet(one, index=False)
    assert main([str(one), "--candidate", f"new={two}", "--config", str(config), "--output", str(output)]) == 0
    summary = pd.read_csv(output / "summary.csv")
    assert summary.lead.tolist() == [1, 1]
