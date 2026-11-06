import io
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, ttest_1samp

from serein.eval.comparison import (
    compare_predictions,
    diebold_mariano,
    holm_adjust,
    main,
)


def prediction_table():
    dates = pd.date_range("2023-08-01", periods=12)
    return pd.DataFrame({
        "issue_date": np.repeat(dates, 2), "point_id": ["A", "B"] * len(dates),
        "y_L1": 0., "smokecast_L1": np.tile([1., 2., 3., 2.], 6),
        "noaa_best_L1": np.repeat(np.arange(1., 13.), 2),
        "persistence_L1": np.tile([2., 3., 2., 1.], 6),
    })


def test_one_day_horizon_agrees_with_independent_one_sample_test():
    difference = np.array([-4., -1., -2., 0., 1., -3.])
    result = diebold_mariano(difference, pd.date_range("2023-08-01", periods=6))
    t_stat = ttest_1samp(difference, 0).statistic
    expected = t_stat * np.sqrt(len(difference) / (len(difference) - 1))
    assert result["status"] == "ok"
    assert result["dm_stat"] == pytest.approx(expected)
    assert result["p_two_sided"] == pytest.approx(2 * norm.cdf(expected))


@pytest.mark.parametrize("gapped", [False, True])
@pytest.mark.parametrize("max_lag", [0, 1, 2, 4])
def test_hac_matches_calendar_kernel_quadratic_form(max_lag, gapped):
    difference = np.array([-3., -2., 1., 2., -1., 3., 2., -4., -2., 1.])
    offsets = np.arange(len(difference))
    if gapped:
        offsets[4:] += 7
    dates = pd.Timestamp("2023-08-01") + pd.to_timedelta(offsets, unit="D")
    centered = difference - difference.mean()
    distances = np.abs(offsets[:, None] - offsets[None, :])
    kernel = np.maximum(0, 1 - distances / (max_lag + 1))
    variance_of_mean = centered @ kernel @ centered / len(difference) ** 2
    expected = difference.mean() / np.sqrt(variance_of_mean)
    result = diebold_mariano(difference, dates, max_lag=max_lag)
    assert result["dm_stat"] == pytest.approx(expected)
    assert result["p_two_sided"] == pytest.approx(2 * norm.sf(abs(expected)))


def test_missing_week_is_not_compressed_into_adjacent_days():
    difference = [-4., -3., -2., 4., 3., 2.]
    consecutive = pd.date_range("2023-08-01", periods=6)
    gapped = consecutive + pd.to_timedelta([0, 0, 0, 7, 7, 7], unit="D")
    # Nonzero mean is needed to expose a different standard error in the statistic.
    difference = np.asarray(difference) + 1
    gap_test = diebold_mariano(difference, gapped, max_lag=1)
    regular_test = diebold_mariano(difference, consecutive, max_lag=1)
    assert gap_test["dm_stat"] != pytest.approx(regular_test["dm_stat"])


def test_dm_is_invariant_to_order_and_positive_units_and_reverses_sign():
    difference = np.array([-4., 3., -2., 1., -3., 2.])
    dates = pd.date_range("2023-08-01", periods=6)
    original = diebold_mariano(difference, dates, max_lag=2)
    order = [4, 1, 3, 5, 0, 2]
    shuffled = diebold_mariano(difference[order] * 1000, dates[order], max_lag=2)
    reversed_test = diebold_mariano(-difference, dates, max_lag=2)
    assert shuffled["dm_stat"] == pytest.approx(original["dm_stat"])
    assert shuffled["p_two_sided"] == pytest.approx(original["p_two_sided"])
    assert reversed_test["dm_stat"] == pytest.approx(-original["dm_stat"])
    assert reversed_test["p_two_sided"] == pytest.approx(original["p_two_sided"])


@pytest.mark.parametrize("difference", [np.zeros(8), np.ones(8), -np.ones(8)])
def test_degenerate_losses_do_not_claim_significance(difference):
    result = diebold_mariano(difference, pd.date_range("2023-08-01", periods=8))
    assert result["status"] == "zero_variance"
    assert np.isnan(result["dm_stat"])
    assert np.isnan(result["p_two_sided"])


@pytest.mark.parametrize("n,max_lag", [(0, 0), (1, 0), (4, 0), (6, 5)])
def test_short_series_has_explicit_unavailable_status(n, max_lag):
    result = diebold_mariano(np.arange(n), pd.date_range("2023-08-01", periods=n),
                            max_lag=max_lag)
    assert result["status"] == "insufficient_days"
    assert np.isnan(result["p_two_sided"])
    assert result["n_days"] == n


@pytest.mark.parametrize("dates", [
    ["2023-08-01"] * 6,
    ["2023-08-01", None, "2023-08-03", "2023-08-04", "2023-08-05", "2023-08-06"],
    pd.date_range("2023-08-01", periods=6, tz="UTC"),
    pd.date_range("2023-08-01 12:00", periods=6),
    np.arange(6),
])
def test_dm_rejects_ambiguous_or_duplicate_dates(dates):
    with pytest.raises(ValueError, match="dates"):
        diebold_mariano(np.arange(6), dates)


@pytest.mark.parametrize("difference", [np.ones((6, 1)), np.ones(5),
                                         [1., 2., 3., 4., 5., np.nan],
                                         [1., 2., 3., 4., 5., np.inf]])
def test_dm_rejects_invalid_loss_series(difference):
    with pytest.raises(ValueError, match="shape|finite"):
        diebold_mariano(difference, pd.date_range("2023-08-01", periods=6))


@pytest.mark.parametrize("max_lag", [-1, 1.5, True, np.nan])
def test_dm_requires_integer_calendar_bandwidth(max_lag):
    with pytest.raises(ValueError, match="max_lag"):
        diebold_mariano(np.arange(6), pd.date_range("2023-08-01", periods=6), max_lag)


def test_holm_adjustment_preserves_order_ties_and_unavailable_comparisons():
    p = [0.03, np.nan, 0.01, 0.01, 0.8]
    # Five planned tests, including the unavailable one. Ties get the running maximum.
    np.testing.assert_allclose(holm_adjust(p), [0.09, np.nan, 0.05, 0.05, 1.])
    assert np.asarray(holm_adjust([])).size == 0
    assert np.isnan(holm_adjust([np.nan])).all()


@pytest.mark.parametrize("p", [[-0.01], [1.01], [np.inf], [[0.5]]])
def test_holm_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust(p)


def test_comparison_uses_equal_day_losses_and_paired_site_day_rmse():
    table = prediction_table()
    table.loc[0, "smokecast_L1"] = np.nan
    table.loc[1, "noaa_best_L1"] = np.inf
    table.loc[2, "y_L1"] = np.nan
    untouched = table.copy(deep=True)
    summary, daily = compare_predictions(table, baselines=["noaa_best"])
    pd.testing.assert_frame_equal(table, untouched)
    row = summary.iloc[0]
    paired = table.dropna(subset=["smokecast_L1", "noaa_best_L1", "y_L1"])
    paired = paired[np.isfinite(paired.noaa_best_L1)]
    difference = paired.smokecast_L1 ** 2 - paired.noaa_best_L1 ** 2
    expected_mean = difference.groupby(paired.issue_date).mean().mean()
    assert row.n_paired == len(paired)
    assert row.n_excluded == len(table) - len(paired)
    assert row.n_days == paired.issue_date.nunique()
    assert row.mean_loss_diff == pytest.approx(expected_mean)
    assert row.rmse_model == pytest.approx(np.sqrt((paired.smokecast_L1 ** 2).mean()))
    assert row.rmse_baseline == pytest.approx(np.sqrt((paired.noaa_best_L1 ** 2).mean()))
    assert daily.n_paired.sum() == row.n_paired
    assert daily.loss_diff.mean() == pytest.approx(row.mean_loss_diff)
    assert row.rmse_diff == pytest.approx(row.rmse_model - row.rmse_baseline)
    assert row.bootstrap_status == "ok"


def test_bootstrap_preserves_a_known_uniform_rmse_improvement():
    table = prediction_table()
    table["smokecast_L1"], table["noaa_best_L1"] = 1., 2.
    summary, _ = compare_predictions(table, baselines=["noaa_best"], n_boot=50, seed=7)
    row = summary.iloc[0]
    assert row.rmse_diff == -1.
    assert row.rmse_diff_ci_lo == -1.
    assert row.rmse_diff_ci_hi == -1.
    assert row.n_boot == 50
    assert row.bootstrap_seed == 7
    # A constant loss difference has no estimated DM variance, even though the
    # empirical resampling interval is a point.
    assert row.status == "zero_variance"
    assert np.isnan(row.p_two_sided)


def test_bootstrap_is_reproducible_and_site_replication_invariant():
    table = prediction_table()
    first, _ = compare_predictions(table, n_boot=100, seed=8)
    again, _ = compare_predictions(table, n_boot=100, seed=8)
    pd.testing.assert_frame_equal(first, again)
    replicated = pd.concat([table.assign(point_id=table.point_id + str(i)) for i in range(3)])
    repeated, _ = compare_predictions(replicated, n_boot=100, seed=8)
    np.testing.assert_allclose(first.rmse_diff_ci_lo, repeated.rmse_diff_ci_lo)
    np.testing.assert_allclose(first.rmse_diff_ci_hi, repeated.rmse_diff_ci_hi)


def test_short_series_keeps_rmse_estimate_with_unavailable_intervals():
    table = prediction_table().iloc[:4]
    summary, _ = compare_predictions(table, baselines=["noaa_best"])
    row = summary.iloc[0]
    assert np.isfinite(row.rmse_diff)
    assert row.bootstrap_status == "insufficient_days"
    assert np.isnan(row.rmse_diff_ci_lo)
    assert np.isnan(row.rmse_diff_ci_hi)


@pytest.mark.parametrize("options", [{"n_boot": 0}, {"n_boot": -1}, {"n_boot": True},
                                     {"n_boot": 2.5}, {"seed": -1}, {"seed": True}])
def test_bootstrap_settings_are_validated_even_with_no_paired_rows(options):
    table = prediction_table()
    table["smokecast_L1"] = np.nan
    with pytest.raises(ValueError, match="n_boot|seed"):
        compare_predictions(table, **options)


def test_identical_replication_of_sites_does_not_inflate_day_sample_size():
    table = prediction_table()
    summary, _ = compare_predictions(table)
    repeated = pd.concat([table.assign(point_id=table.point_id + str(i)) for i in range(10)])
    replicated, _ = compare_predictions(repeated)
    np.testing.assert_allclose(summary.dm_stat, replicated.dm_stat, equal_nan=True)
    np.testing.assert_allclose(summary.p_two_sided, replicated.p_two_sided, equal_nan=True)
    np.testing.assert_array_equal(summary.n_days, replicated.n_days)
    np.testing.assert_array_equal(summary.n_paired * 10, replicated.n_paired)


def test_comparison_reports_each_requested_baseline_and_lead_in_one_family():
    table = prediction_table()
    for name in ("y", "smokecast", "noaa_best", "persistence"):
        table[f"{name}_L3"] = table[f"{name}_L1"] * 2
        table[f"{name}_L0"] = table[f"{name}_L1"]
    summary, daily = compare_predictions(table)
    assert list(zip(summary.lead, summary.baseline)) == [
        (lead, baseline) for lead in (0, 1, 3) for baseline in ("noaa_best", "persistence")
    ]
    np.testing.assert_array_equal(summary.max_lag, [0, 0, 0, 0, 2, 2])
    np.testing.assert_allclose(summary.p_two_sided_holm, holm_adjust(summary.p_two_sided))
    assert set(daily.lead) == {0, 1, 3}


def test_comparison_keeps_a_baseline_with_no_paired_rows():
    table = prediction_table()
    table["noaa_best_L1"] = np.nan
    summary, daily = compare_predictions(table)
    row = summary[summary.baseline == "noaa_best"].iloc[0]
    assert row.n_paired == 0
    assert row.n_excluded == len(table)
    assert row.status == "insufficient_days"
    assert np.isnan(row.p_two_sided_holm)
    audit = daily[daily.baseline == "noaa_best"]
    assert len(audit) == table.issue_date.nunique()
    assert audit.n_paired.eq(0).all()
    assert audit.n_excluded.sum() == len(table)
    assert audit.loss_diff.isna().all()


def test_comparison_retains_days_lost_to_paired_missingness():
    table = prediction_table()
    table.loc[:1, "smokecast_L1"] = np.nan
    summary, daily = compare_predictions(table, baselines=["noaa_best"])
    assert summary.iloc[0].n_days_excluded == 1
    assert summary.iloc[0].n_days_input == table.issue_date.nunique()
    first = daily.iloc[0]
    assert first.n_paired == 0
    assert first.n_excluded == first.n_input == 2
    assert np.isnan(first.loss_diff)


def test_nullable_missing_predictions_use_the_same_paired_mask():
    table = prediction_table()
    table["smokecast_L1"] = table.smokecast_L1.astype("Float64")
    table.loc[0, "smokecast_L1"] = pd.NA
    result, _ = compare_predictions(table)
    assert result.n_excluded.eq(1).all()


def test_dm_large_loss_scale_does_not_overflow_mean_or_variance():
    difference = np.array([1., 2., 3., 4., 5., 6.])
    dates = pd.date_range("2023-08-01", periods=6)
    expected = diebold_mariano(difference, dates, max_lag=1)
    result = diebold_mariano(difference * 1e307, dates, max_lag=1)
    assert np.isfinite(result["mean_loss_diff"])
    assert result["mean_loss_diff"] == pytest.approx(expected["mean_loss_diff"] * 1e307)
    assert result["dm_stat"] == pytest.approx(expected["dm_stat"])


def test_comparison_large_finite_losses_produce_finite_rmse():
    table = prediction_table()
    table["smokecast_L1"] = 1e154
    summary, _ = compare_predictions(table, baselines=["noaa_best"])
    assert summary.iloc[0].rmse_model == pytest.approx(1e154)


def test_comparison_rejects_duplicate_site_day_before_scoring():
    table = prediction_table()
    with pytest.raises(ValueError, match="unique.*point_id"):
        compare_predictions(pd.concat([table, table.iloc[:1]]))


@pytest.mark.parametrize("point_id", [None, "", "  ", 42])
def test_comparison_rejects_missing_or_invalid_site_identifiers(point_id):
    table = prediction_table()
    table["point_id"] = table.point_id.astype(object)
    table.loc[0, "point_id"] = point_id
    with pytest.raises(ValueError, match="point_id"):
        compare_predictions(table)


@pytest.mark.parametrize("options", [
    {"leads": []}, {"leads": [1, 1]}, {"leads": [True]}, {"leads": [-1]},
    {"baselines": []}, {"baselines": ["noaa_best", "noaa_best"]},
    {"baselines": ["smokecast"]}, {"leads": [2]}, {"baselines": ["absent"]},
])
def test_comparison_rejects_invalid_or_incomplete_comparisons(options):
    with pytest.raises(ValueError):
        compare_predictions(prediction_table(), **options)


def test_default_baseline_with_incomplete_lead_columns_is_not_silently_omitted():
    table = prediction_table()
    table["y_L2"] = 0.
    table["smokecast_L2"] = 1.
    table["noaa_best_L2"] = 2.
    with pytest.raises(ValueError, match="persistence_L2"):
        compare_predictions(table)


def test_smoke_subset_uses_only_known_plumes_and_reports_eligibility():
    table = prediction_table()
    table["hms_prev"] = np.tile([0., 1., 2., 3., np.nan, 0.], 4)
    summary, daily = compare_predictions(table, subsets=["smoke_yesterday"], n_boot=50)
    expected = table[table.hms_prev >= 1]
    assert summary.subset.eq("smoke_yesterday").all()
    assert summary.n_total.eq(len(table)).all()
    assert summary.n_input.eq(len(expected)).all()
    assert summary.n_subset_excluded.eq(len(table) - len(expected)).all()
    assert summary.n_subset_unknown.eq(table.hms_prev.isna().sum()).all()
    noaa = summary[summary.baseline == "noaa_best"].iloc[0]
    assert noaa.rmse_model == pytest.approx(np.sqrt((expected.smokecast_L1 ** 2).mean()))
    noaa_audit = daily[daily.baseline == "noaa_best"]
    assert noaa_audit.n_input.sum() == len(expected)
    assert set(noaa_audit.issue_date) == set(expected.issue_date)


def test_multiple_subsets_share_a_single_multiplicity_family():
    table = prediction_table()
    table["hms_prev"] = np.tile([0., 3.], len(table) // 2)
    summary, daily = compare_predictions(table, subsets=["all", "smoke_yesterday"], n_boot=50)
    assert summary.family_size.eq(len(summary)).all()
    np.testing.assert_allclose(summary.p_two_sided_holm, holm_adjust(summary.p_two_sided))
    assert set(daily.subset) == {"all", "smoke_yesterday"}
    for subset in ("all", "smoke_yesterday"):
        alone, _ = compare_predictions(table, subsets=[subset], n_boot=50)
        joint = summary[summary.subset == subset]
        np.testing.assert_allclose(joint.dm_stat, alone.dm_stat, equal_nan=True)
        np.testing.assert_allclose(joint.rmse_diff_ci_lo, alone.rmse_diff_ci_lo)
        assert (joint.p_two_sided_holm.to_numpy() >= alone.p_two_sided_holm.to_numpy()).all()


def test_subset_selection_generators_are_reused_for_each_subset():
    table = prediction_table()
    table["hms_prev"] = 3.
    summary, _ = compare_predictions(
        table, subsets=iter(["all", "smoke_yesterday"]), leads=iter([1]),
        baselines=iter(["noaa_best", "persistence"]), n_boot=10,
    )
    assert len(summary) == 4
    assert summary.family_size.eq(4).all()


def test_empty_smoke_subset_keeps_unavailable_comparisons_in_the_family():
    table = prediction_table()
    table["hms_prev"] = 0.
    summary, daily = compare_predictions(table, subsets=["all", "smoke_yesterday"], n_boot=20)
    empty = summary[summary.subset == "smoke_yesterday"]
    assert len(empty) == 2
    assert empty.n_input.eq(0).all()
    assert empty.n_subset_excluded.eq(len(table)).all()
    assert empty.status.eq("insufficient_days").all()
    assert empty.p_two_sided_holm.isna().all()
    assert summary.family_size.eq(4).all()
    assert not daily.subset.eq("smoke_yesterday").any()


@pytest.mark.parametrize("density", [-1., 4., 0.5, np.inf, -np.inf, "bad"])
def test_smoke_subset_rejects_invalid_hms_density(density):
    table = prediction_table()
    table["hms_prev"] = pd.Series([density] * len(table), dtype=object)
    with pytest.raises(ValueError):
        compare_predictions(table, subsets=["smoke_yesterday"])


def test_all_day_comparison_does_not_require_or_interpret_hms():
    table = prediction_table()
    table["hms_prev"] = "not used"
    summary, _ = compare_predictions(table, n_boot=10)
    assert summary.n_subset_excluded.eq(0).all()
    assert summary.n_subset_unknown.eq(0).all()
    with pytest.raises(ValueError, match="hms_prev"):
        compare_predictions(table.drop(columns="hms_prev"), subsets=["smoke_yesterday"])


@pytest.mark.parametrize("subsets", [[], ["smoke"], ["all", "all"]])
def test_invalid_subset_selection_is_rejected(subsets):
    with pytest.raises(ValueError, match="subsets"):
        compare_predictions(prediction_table(), subsets=subsets)


def test_duplicates_cannot_be_hidden_by_smoke_filtering():
    table = prediction_table()
    table["hms_prev"] = 0.
    table = pd.concat([table, table.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        compare_predictions(table, subsets=["smoke_yesterday"])


def test_common_rows_equalize_populations_and_expose_extra_attrition():
    table = prediction_table()
    table.loc[0, "noaa_best_L1"] = np.nan
    table.loc[2:3, "persistence_L1"] = np.nan
    table.loc[4, "smokecast_L1"] = np.nan
    pairwise, _ = compare_predictions(table, n_boot=50)
    common, daily = compare_predictions(table, common_rows=True, n_boot=50)
    expected = table.dropna(subset=["noaa_best_L1", "persistence_L1", "smokecast_L1", "y_L1"])
    assert pairwise.n_paired.nunique() > 1
    assert common.n_paired.eq(len(expected)).all()
    assert common.population.eq("common_rows").all()
    assert common.rmse_model.nunique() == 1
    np.testing.assert_array_equal(common.n_pairwise_available, pairwise.n_paired)
    np.testing.assert_array_equal(common.n_excluded_common, pairwise.n_paired - common.n_paired)
    for baseline in common.baseline:
        audit = daily[daily.baseline == baseline]
        assert audit.n_paired.sum() == len(expected)
        assert audit.n_paired.eq(0).sum() == common.iloc[0].n_days_excluded


def test_common_population_is_computed_separately_at_each_lead():
    table = prediction_table()
    for name in ("y", "smokecast", "noaa_best", "persistence"):
        table[f"{name}_L2"] = table[f"{name}_L1"]
    table.loc[0, "noaa_best_L1"] = np.nan
    table.loc[2, "persistence_L2"] = np.nan
    _, daily = compare_predictions(table, common_rows=True, n_boot=20)
    one = daily[(daily.lead == 1) & (daily.baseline == "noaa_best")]
    two = daily[(daily.lead == 2) & (daily.baseline == "noaa_best")]
    assert one.n_paired.iloc[:2].tolist() == [1, 2]
    assert two.n_paired.iloc[:2].tolist() == [2, 1]


def test_unavailable_baseline_makes_common_population_explicitly_empty():
    table = prediction_table()
    table["noaa_best_L1"] = np.nan
    summary, daily = compare_predictions(table, common_rows=True, n_boot=10)
    assert summary.n_paired.eq(0).all()
    assert summary.status.eq("insufficient_days").all()
    assert daily.n_paired.eq(0).all()
    assert summary[summary.baseline == "persistence"].iloc[0].n_excluded_common == len(table)


def test_common_rows_option_requires_a_boolean():
    with pytest.raises(TypeError, match="common_rows"):
        compare_predictions(prediction_table(), common_rows=1)


def test_common_population_and_smoke_subset_combine(tmp_path, capsys):
    table = prediction_table()
    table["hms_prev"] = np.tile([0., 3.], len(table) // 2)
    table.loc[1, "noaa_best_L1"] = np.nan
    source = tmp_path / "predictions.parquet"
    table.to_parquet(source)
    assert main([str(source), "--subset", "all", "--subset", "smoke_yesterday",
                 "--common-rows", "--n-boot", "20", "--seed", "3"]) == 0
    printed = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert set(printed.subset) == {"all", "smoke_yesterday"}
    assert printed.population.eq("common_rows").all()
    assert printed.family_size.eq(len(printed)).all()
    for subset in printed.subset.unique():
        assert printed[printed.subset == subset].n_paired.nunique() == 1


def test_comparison_refuses_overflow_instead_of_reporting_infinite_loss():
    table = prediction_table()
    table["smokecast_L1"] = 1e300
    with pytest.raises(ValueError, match="squared forecast errors"):
        compare_predictions(table)


def test_cli_reads_parquet_and_exports_reproducible_summary_and_audit(tmp_path):
    table = prediction_table()
    table.to_parquet(tmp_path / "predictions.parquet", index=False)
    summary_path, daily_path = tmp_path / "summary.csv", tmp_path / "daily.csv"
    completed = subprocess.run(
        [sys.executable, "-m", "serein.eval.comparison", str(tmp_path),
         "--baseline", "noaa_best", "--lead", "1", "--max-lag", "2",
         "--output", str(summary_path), "--daily-output", str(daily_path)],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout == ""
    expected, daily = compare_predictions(table, baselines=["noaa_best"], leads=[1], max_lag=2)
    pd.testing.assert_frame_equal(pd.read_csv(summary_path), expected)
    pd.testing.assert_frame_equal(pd.read_csv(daily_path, parse_dates=["issue_date"]), daily)


def test_cli_defaults_to_stdout_without_writing_reports(tmp_path, capsys):
    source = tmp_path / "predictions.parquet"
    prediction_table().to_parquet(source)
    assert main([str(source), "--baseline", "noaa_best"]) == 0
    printed = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert printed.baseline.tolist() == ["noaa_best"]
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("existing", ["summary", "daily"])
def test_cli_refuses_existing_outputs_before_writing_any_report(tmp_path, existing):
    source = tmp_path / "predictions.parquet"
    prediction_table().to_parquet(source)
    paths = {"summary": tmp_path / "summary.csv", "daily": tmp_path / "daily.csv"}
    paths[existing].write_text("keep this file", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main([str(source), "--output", str(paths["summary"]),
              "--daily-output", str(paths["daily"])])
    assert error.value.code == 2
    assert paths[existing].read_text(encoding="utf-8") == "keep this file"
    assert not paths["daily" if existing == "summary" else "summary"].exists()


def test_cli_rejects_shared_output_path(tmp_path):
    source = tmp_path / "predictions.parquet"
    prediction_table().to_parquet(source)
    target = tmp_path / "shared.csv"
    with pytest.raises(SystemExit) as error:
        main([str(source), "--output", str(target), "--daily-output", str(target)])
    assert error.value.code == 2
    assert not target.exists()
