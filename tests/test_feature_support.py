from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp, wasserstein_distance

from serein.eval.feature_support import analyze_support, main, write_support


def table(values, *, start="2023-06-01", point="001"):
    return pd.DataFrame({"point_id": point, "issue_date": pd.date_range(start, periods=len(values)),
                         "aqm_mean_L1": values})


def analyze(reference, query, **kwargs):
    return analyze_support(reference, query, features=["aqm_mean_L1"], **kwargs)


def test_exact_disjoint_distributions_and_row_flags():
    ref, query = table([0., 2., 4., np.nan]), table([5., 6., np.nan])
    result = analyze(ref, query)
    summary = result["summary"].iloc[0]
    shift = result["shift"].iloc[0]
    assert shift.ks_distance == 1
    assert shift.wasserstein == pytest.approx(3.5)
    assert shift.query_mean == 5.5
    assert shift.query_n_finite == 2
    assert shift.n_reference_finite == 3
    assert shift.n_reference_finite_days == 3
    assert shift.n_query_finite_days == 2
    assert shift.missing_rate_change == pytest.approx(1 / 3 - 1 / 4)
    assert shift.above_reference_max_rate == 1
    assert shift.below_reference_min_rate == 0
    assert shift.outside_reference_90_rate == 1
    assert shift.mean_shift_ref_sd == pytest.approx(3.5 / np.std([0, 2, 4]))
    assert shift.wasserstein_ref_sd == pytest.approx(3.5 / np.std([0, 2, 4]))
    assert result["rows"].n_outside_reference_range.tolist() == [1, 1, 0]
    assert result["rows"].n_missing_features.tolist() == [0, 0, 1]
    assert summary.n_rows_outside_reference_range == 2
    assert summary.n_rows_with_missing_features == 1


@pytest.mark.parametrize("a,b", [
    ([0, 0, 1], [0, 0, 1]), ([0, 0, 0], [0, 1, 1]),
    ([0, 0, 2, 5, 5], [-1, 0, 2, 2, 5, 8]), ([4], [4]),
    ([-4, 0, 4], [0, 0, 0]), ([1, 2], [1]), ([1], [1, 2]),
])
def test_tied_empirical_cdfs_match_scipy(a, b):
    result = analyze(table(a), table(b))["shift"].iloc[0]
    assert result.ks_distance == pytest.approx(ks_2samp(a, b).statistic)
    assert result.wasserstein == pytest.approx(wasserstein_distance(a, b))


@pytest.mark.parametrize("seed", range(8))
def test_random_distributions_match_independent_library(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(-20, 20, 61), rng.integers(-10, 30, 37)
    actual = analyze(table(a), table(b))["shift"].iloc[0]
    assert actual.ks_distance == pytest.approx(ks_2samp(a, b).statistic)
    assert actual.wasserstein == pytest.approx(wasserstein_distance(a, b))


def test_equal_distributions_including_ties_have_zero_distance():
    result = analyze(table([0, 1, 1, 2]), table([2, 1, 0, 1]))
    row = result["shift"].iloc[0]
    assert row.ks_distance == row.wasserstein == row.mean_shift_ref_sd == 0
    assert row.below_reference_min_rate == row.above_reference_max_rate == 0
    # Quantile tails are strict; equality with an extreme is not outside the range.
    assert result["rows"].n_outside_reference_range.sum() == 0


@pytest.mark.parametrize("query", [[3, 3], [3, 4], [1, 2]])
def test_constant_reference_retains_raw_distances_without_infinite_standardization(query):
    result = analyze(table([3, 3]), table(query))["shift"].iloc[0]
    assert result.status == "constant_reference"
    assert np.isnan(result.mean_shift_ref_sd)
    assert np.isnan(result.wasserstein_ref_sd)
    assert result.wasserstein == pytest.approx(wasserstein_distance([3, 3], query))


@pytest.mark.parametrize("a,b,status", [
    ([np.nan, np.nan], [1, 2], "no_reference_values"),
    ([1, 2], [np.nan, np.nan], "no_query_values"),
    ([np.nan], [np.nan], "no_reference_values"),
])
def test_all_missing_inputs_remain_visible(a, b, status):
    result = analyze(table(a), table(b))
    shift = result["shift"].iloc[0]
    assert shift.status == status
    assert np.isnan(shift.ks_distance)
    assert np.isnan(shift.wasserstein)
    assert result["summary"].iloc[0].n_rows == len(b)
    assert result["rows"].n_features_without_reference.tolist() == [int(status == "no_reference_values")] * len(b)


def test_point_scope_exposes_unseen_and_locally_unfamiliar_values():
    reference = pd.concat([table([0, 1], point="001"), table([10, 11], point="002")])
    query = pd.concat([table([10], point="001"), table([10], point="002"), table([5], point="003")])
    pooled = analyze(reference, query)
    local = analyze(reference, query, reference_scope="point", groups=["point", "month"])
    assert pooled["rows"].n_outside_reference_range.tolist() == [0, 0, 0]
    assert local["rows"].n_outside_reference_range.tolist() == [1, 0, 0]
    assert local["rows"].seen_reference_point.tolist() == [True, True, False]
    assert local["rows"].n_features_without_reference.tolist() == [0, 0, 1]
    assert local["summary"].iloc[0].n_unseen_point_rows == 1
    assert local["summary"].iloc[0].n_rows_without_reference == 1
    unseen = local["shift"].query("reference_point == '003'")
    assert len(unseen) == 3
    assert unseen.status.eq("no_reference_values").all()
    assert unseen.n_reference_rows.eq(0).all()
    assert unseen.n_reference_days.eq(0).all()
    known = local["shift"].query("reference_point == '001'")
    assert known.ks_distance.eq(1).all()


def test_month_groups_do_not_compare_against_a_query_fitted_reference():
    reference = table([1, 2, 3])
    query = table([1, 2, 9, 10], start="2023-06-29")
    result = analyze(reference, query, groups=["month"])
    rows = result["shift"].set_index("group_value")
    assert rows.loc["2023-06", "n_reference_rows"] == 3
    assert rows.loc["2023-07", "n_reference_rows"] == 3
    assert rows.loc["2023-07", "ks_distance"] == 1
    assert rows.loc["2023-06", "query_mean"] == 1.5
    assert rows.loc["2023-07", "query_mean"] == 9.5


def test_multiple_features_counts_use_any_feature_per_row():
    reference, query = table([0, 2]), table([4., np.nan, 1.])
    reference["hms_prev"] = [0., 1.]
    query["hms_prev"] = [3., 0., np.nan]
    result = analyze_support(reference, query, features=["aqm_mean_L1", "hms_prev"])
    assert result["rows"].n_outside_reference_range.tolist() == [2, 0, 0]
    assert result["rows"].n_missing_features.tolist() == [0, 1, 1]
    assert result["summary"].iloc[0].n_rows_outside_reference_range == 1
    assert result["summary"].iloc[0].n_rows_with_missing_features == 2


def test_inputs_targets_and_indices_do_not_affect_audit():
    reference, query = table([1, 2, 3]), table([3, 2, 1])
    reference.index = query.index = [7, 7, 7]
    before = reference.copy(deep=True), query.copy(deep=True)
    expected = analyze(reference, query)
    pd.testing.assert_frame_equal(reference, before[0])
    pd.testing.assert_frame_equal(query, before[1])
    reference["y_L0"] = [np.inf, np.nan, -np.inf]
    query["y_L1"] = ["not", "an", "input"]
    actual = analyze(reference, query)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])


@pytest.mark.parametrize("scope", ["pooled", "point"])
def test_permutation_and_nullable_inputs(scope):
    reference = pd.concat([table([0, 2], point="001"), table([7, 9], point="002")], ignore_index=True)
    query = pd.concat([table([0., np.nan], point="001"), table([4., 11.], point="002")], ignore_index=True)
    reference["aqm_mean_L1"] = reference.aqm_mean_L1.astype("Int64")
    query["aqm_mean_L1"] = query.aqm_mean_L1.astype("Float64")
    a = analyze(reference, query, reference_scope=scope, groups=["point", "month"])
    b = analyze(reference.sample(frac=1, random_state=1), query.sample(frac=1, random_state=2),
                reference_scope=scope, groups=["point", "month"])
    for name in a:
        keys = (["point_id", "issue_date"] if name == "rows" else list(a[name].columns[:4]))
        pd.testing.assert_frame_equal(a[name].sort_values(keys).reset_index(drop=True),
                                      b[name].sort_values(keys).reset_index(drop=True))


@pytest.mark.parametrize("which", ["reference", "query"])
@pytest.mark.parametrize("bad", ["empty", "duplicate", "id", "date", "timezone", "reserved",
                                "missing_column", "text", "infinity", "complex"])
def test_malformed_tables_rejected(which, bad):
    frames = {"reference": table([1., 2.]), "query": table([3., 4.])}
    frame = frames[which]
    if bad == "empty":
        frames[which] = frame.iloc[:0]
    elif bad == "duplicate":
        frames[which] = pd.concat([frame, frame.iloc[:1]])
    elif bad == "id":
        frame["point_id"] = ["", "002"]
    elif bad == "date":
        frame["issue_date"] = ["invalid", "2023-06-02"]
    elif bad == "timezone":
        frame["issue_date"] = frame.issue_date.dt.tz_localize("UTC")
    elif bad == "reserved":
        frame["issue_date"] = pd.date_range("2024-01-01", periods=2)
    elif bad == "missing_column":
        frames[which] = frame.drop(columns="aqm_mean_L1")
    elif bad == "text":
        frame["aqm_mean_L1"] = ["1", "2"]
    elif bad == "infinity":
        frame.loc[0, "aqm_mean_L1"] = np.inf
    else:
        frame["aqm_mean_L1"] = [1 + 2j, 2 + 3j]
    with pytest.raises((ValueError, TypeError)):
        analyze(**frames)


@pytest.mark.parametrize("features", [[], ["y_L0"], ["point_id"], ["kind"], ["issue_date"],
                                     ["a", "a"], ["a b"], [None], [["a"]], "aqm_mean_L1"])
def test_feature_selection_is_explicit_and_cannot_use_targets(features):
    with pytest.raises((ValueError, TypeError)):
        analyze_support(table([1]), table([1]), features=features)


@pytest.mark.parametrize("kwargs", [{"reference_scope": "other"}, {"groups": ["smoke"]},
                                    {"groups": ["point", "point"]}])
def test_invalid_grouping_rejected(kwargs):
    with pytest.raises(ValueError):
        analyze(table([1]), table([1]), **kwargs)


def test_overflow_is_not_silently_a_support_score():
    with pytest.raises(ValueError, match="overflow"):
        analyze(table([-1e308, 1e308]), table([1.]))


def test_writer_hashes_decoded_bytes_and_preserves_inputs(tmp_path):
    ref, query, output = tmp_path / "reference.parquet", tmp_path / "query.parquet", tmp_path / "report"
    table([0, 1]).to_parquet(ref)
    table([2, 3], start="2023-06-02").to_parquet(query)
    original = {path: path.read_bytes() for path in (ref, query)}
    manifest = write_support(ref, query, output, features=["aqm_mean_L1"], groups=["point"])
    assert manifest["overlapping_point_issue_rows"] == 1
    for name, path in (("reference", ref), ("query", query)):
        assert path.read_bytes() == original[path]
        assert manifest["sources"][name]["sha256"] == hashlib.sha256(original[path]).hexdigest()
    assert json.loads((output / "manifest.json").read_text()) == manifest
    rows = pd.read_parquet(output / "rows.parquet")
    assert rows.point_id.tolist() == ["001", "001"]
    assert rows.n_outside_reference_range.tolist() == [1, 1]
    assert {path.name for path in output.iterdir()} == {
        "reference.csv", "shift.csv", "summary.csv", "rows.parquet", "manifest.json"}
    with pytest.raises(FileExistsError):
        write_support(ref, query, output, features=["aqm_mean_L1"])
    assert pd.read_parquet(output / "rows.parquet").equals(rows)


def test_invalid_input_does_not_create_output(tmp_path):
    source = tmp_path / "table.parquet"
    table([np.inf]).to_parquet(source)
    output = tmp_path / "report"
    with pytest.raises(ValueError):
        write_support(source, source, output, features=["aqm_mean_L1"])
    assert not output.exists()


def test_final_run_lock_refused_before_decoding(tmp_path):
    (tmp_path / "FINAL_RUN_LOCK.json").write_text("{}")
    with pytest.raises(ValueError, match="development"):
        write_support(tmp_path / "absent.parquet", tmp_path / "absent.parquet", tmp_path / "report",
                      features=["aqm_mean_L1"])


def test_cli_success_and_actionable_failure(tmp_path, capsys):
    source = tmp_path / "table.parquet"
    table([0, 1]).to_parquet(source)
    args = ["--reference", str(source), "--query", str(source), "--out", str(tmp_path / "report"),
            "--feature", "aqm_mean_L1", "--group", "month", "--reference-scope", "point"]
    assert main(args) == 0
    assert "wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit) as failure:
        main(args)
    assert failure.value.code == 2
    assert "output already exists" in capsys.readouterr().err
