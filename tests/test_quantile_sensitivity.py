from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from serein.eval.quantile_sensitivity import (
    DEFAULT_LEVELS,
    analyze_quantile_sensitivity,
    main,
    quantile_probability_bracket,
    write_quantile_sensitivity,
)


def predictions(q=None, p=None, y=None):
    if q is None:
        q = np.array([[5., 10., 20., 40., 60.], [40., 50., 60., 70., 80.],
                      [0., 5., 10., 15., 20.], [np.nan] * 5])
    q = np.asarray(q)
    frame = pd.DataFrame({"point_id": "001", "issue_date": pd.date_range("2023-06-01", periods=len(q)),
                          "y_L1": [0., 50., 50., np.nan] if y is None else y})
    for index in range(q.shape[1]):
        frame[f"smokecast_q{index}_L1"] = q[:, index]
    if p is not None:
        frame["smokecast_prob_L1"] = p
    return frame


def test_brackets_at_interior_and_beyond_tails():
    q = np.array([[5., 10., 20., 40., 60.], [40., 50., 60., 70., 80.], [0., 5., 10., 15., 20.]])
    lower, upper = quantile_probability_bracket(q, DEFAULT_LEVELS)
    np.testing.assert_allclose(lower, [.25, .95, 0.])
    np.testing.assert_allclose(upper, [.5, 1., .05])


def test_equality_and_repeated_quantiles_use_inclusive_event():
    q = np.array([[35.5] * 5, [0., 35.5, 35.5, 35.5, 50.]])
    lower, upper = quantile_probability_bracket(q, DEFAULT_LEVELS)
    np.testing.assert_allclose(lower, [.95, .75])
    np.testing.assert_allclose(upper, [1., .95])


def test_zero_threshold_uses_known_nonnegative_support():
    q = np.array([[0.] * 5, [1., 2., 3., 4., 5.], [np.nan] * 5])
    lower, upper = quantile_probability_bracket(q, DEFAULT_LEVELS, threshold=0)
    np.testing.assert_array_equal(lower[:2], [1., 1.])
    np.testing.assert_array_equal(upper[:2], [1., 1.])
    assert np.isnan(lower[-1]) and np.isnan(upper[-1])


@pytest.mark.parametrize("seed", range(25))
def test_brackets_contain_exact_discrete_distribution_event_probability(seed):
    rng = np.random.default_rng(seed)
    support = np.r_[0., np.sort(rng.uniform(1., 100., 9))]
    weights = rng.dirichlet(np.ones(len(support)))
    cdf = np.cumsum(weights)
    levels = np.r_[.01, .05, .25, .5, .75, .95, .99]
    q = support[np.searchsorted(cdf, levels, side="left")]
    for threshold in np.r_[support, np.nextafter(support, np.inf), 35.5, 120.]:
        lower, upper = quantile_probability_bracket(q, levels, threshold=threshold)
        actual = weights[support >= threshold].sum()
        assert lower <= actual + 1e-14
        assert actual <= upper + 1e-14


def test_dense_axes_empty_arrays_and_no_input_mutation():
    q = np.arange(60.).reshape(3, 4, 5)
    before = q.copy()
    lower, upper = quantile_probability_bracket(q, DEFAULT_LEVELS)
    assert lower.shape == upper.shape == (3, 4)
    np.testing.assert_array_equal(q, before)
    assert quantile_probability_bracket(np.empty((0, 5)), DEFAULT_LEVELS)[0].shape == (0,)
    assert quantile_probability_bracket([20.], [.5])[0] == 0


@pytest.mark.parametrize("q,levels", [
    ([[1., 3., 2.]], [.1, .5, .9]), ([[3., np.nan, 2.]], [.1, .5, .9]),
    ([[1., np.inf, 3.]], [.1, .5, .9]), ([[-1., 2., 3.]], [.1, .5, .9]),
    ([[1., 2.]], [.1, .5, .9]), ([[1., 2.]], [.5, .1]), ([[1., 2.]], [.5, .5]),
    ([[1., 2.]], [0., .5]), ([[1., 2.]], [.5, 1.]), ([[1., 2.]], [.5, np.nan]),
    ([[1., 2.]], [[.1, .5]]), ([[]], []), (1., [.5]),
])
def test_invalid_quantiles_and_levels_are_rejected(q, levels):
    with pytest.raises(ValueError):
        quantile_probability_bracket(q, levels)


@pytest.mark.parametrize("threshold", [-1., np.nan, np.inf, True])
def test_invalid_event_threshold(threshold):
    with pytest.raises(ValueError):
        quantile_probability_bracket([1.], [.5], threshold=threshold)


@pytest.mark.parametrize("complex_values", ["quantiles", "levels"])
def test_direct_array_api_does_not_discard_imaginary_values(complex_values):
    quantiles = np.array([1., 2., 3.])
    levels = np.array([.1, .5, .9])
    if complex_values == "quantiles":
        quantiles = quantiles.astype(complex) + 1j
    else:
        levels = levels.astype(complex) + .1j
    with pytest.raises(ValueError, match="real"):
        quantile_probability_bracket(quantiles, levels)


def test_frame_api_does_not_discard_imaginary_levels():
    levels = np.array(DEFAULT_LEVELS, dtype=complex) + .01j
    with pytest.raises(ValueError, match="real"):
        analyze_quantile_sensitivity(predictions(), levels=levels)


def test_missing_quantiles_leave_brackets_missing_but_detect_known_crossings():
    lower, upper = quantile_probability_bracket([[1., np.nan, 3.], [np.nan, np.nan, np.nan]], [.1, .5, .9])
    assert np.isnan(lower).all() and np.isnan(upper).all()
    with pytest.raises(ValueError, match="nondecreasing"):
        quantile_probability_bracket([[3., np.nan, 1.]], [.1, .5, .9])


def test_fixed_cutoff_ambiguity_counts_and_missingness():
    result = analyze_quantile_sensitivity(predictions(), cutoffs=[.25, .5])
    summary = result["summary"].set_index("cutoff")
    assert summary.n_input.eq(4).all()
    assert summary.n_complete.eq(3).all()
    assert summary.n_incomplete.eq(1).all()
    assert summary.n_observed.eq(3).all()
    assert summary.n_events.eq(2).all()
    assert summary.n_probability.eq(0).all()
    assert summary.n_threshold_above_last.eq(1).all()
    assert summary.n_threshold_below_first.eq(1).all()
    assert summary.loc[.25, ["n_forced_warning", "n_forced_quiet", "n_ambiguous"]].tolist() == [2, 1, 0]
    assert summary.loc[.5, ["n_forced_warning", "n_forced_quiet", "n_ambiguous"]].tolist() == [1, 1, 1]
    assert summary.loc[.5, "n_ambiguous_events"] == 0
    assert summary.loc[.5, "n_forced_quiet_events"] == 1
    assert summary.loc[.5, "ambiguous_fraction"] == pytest.approx(1 / 3)
    rows = result["rows"]
    assert rows.complete_quantiles.tolist() == [True, True, True, False]
    assert np.isnan(rows.probability_lo.iloc[-1])
    assert np.isnan(rows.event.iloc[-1])


def test_bracket_endpoint_cutoffs_and_supplied_probability_disagreement():
    frame = predictions(p=[.6, .2, .1, .5])
    result = analyze_quantile_sensitivity(frame, cutoffs=[0., .05, .25, .5, 1.])
    rows = result["rows"]
    assert rows.probability_outside_bracket.tolist() == [True, True, True, False]
    summary = result["summary"].set_index("cutoff")
    assert summary.n_outside_bracket.eq(3).all()
    assert summary.n_probability.eq(3).all()
    assert summary.loc[0., "n_forced_warning"] == 3
    assert summary.loc[0., "n_warning_disagreement"] == 0
    assert summary.loc[.25, "n_warning_disagreement"] == 1
    assert summary.loc[.5, "n_warning_disagreement"] == 1
    assert summary.loc[1., "n_ambiguous"] == 1
    assert summary.loc[1., "n_forced_quiet"] == 2


def test_collapsed_at_threshold_exposes_atom_convention_without_changing_conversion():
    # A rightmost-tie CDF interpretation yields .05; inclusive P(Y >= t) need not.
    frame = predictions(np.full((4, 5), 35.5), p=[.05] * 4)
    result = analyze_quantile_sensitivity(frame, cutoffs=[.5])
    assert result["rows"].probability_outside_bracket.all()
    assert result["summary"].n_warning_disagreement.tolist() == [4]
    assert result["summary"].n_threshold_equals_quantile.tolist() == [4]
    assert frame.smokecast_prob_L1.eq(.05).all()


def test_optional_conversion_is_audited_without_requiring_targets():
    frame = predictions(p=[.4, np.nan, .025, np.nan], y=[np.nan] * 4)
    summary = analyze_quantile_sensitivity(frame)["summary"]
    assert summary.n_observed.eq(0).all()
    assert summary.n_complete.eq(3).all()
    assert summary.n_probability.eq(2).all()
    assert summary.n_outside_bracket.eq(0).all()


def test_grouped_summaries_reconstruct_from_rows_and_retain_unknown_smoke():
    frame = predictions(p=[.4, .975, .025, np.nan])
    frame["point_id"] = ["001", "001", "002", "002"]
    frame["hms_prev"] = [0., 1., np.nan, 1.]
    result = analyze_quantile_sensitivity(frame, groups=["point", "month", "smoke"])
    assert set(result["summary"].group_value) >= {"001", "002", "2023-06", "smoke", "clear", "unknown"}
    for row in result["summary"].itertuples(index=False):
        assert row.n_forced_warning + row.n_forced_quiet + row.n_ambiguous == row.n_complete
        assert row.n_complete + row.n_incomplete == row.n_input
    shuffled = frame.sample(frac=1, random_state=3)
    shuffled.index = [0] * len(shuffled)
    actual = analyze_quantile_sensitivity(shuffled, groups=["point", "month", "smoke"])
    pd.testing.assert_frame_equal(result["summary"], actual["summary"])
    keys = ["model", "lead", "point_id", "issue_date"]
    pd.testing.assert_frame_equal(result["rows"].sort_values(keys).reset_index(drop=True),
                                  actual["rows"].sort_values(keys).reset_index(drop=True))


def test_subset_models_leads_and_custom_levels():
    frame = predictions()
    frame["other_q0_L2"] = 10.
    frame["other_q1_L2"] = 100.
    frame["y_L2"] = 50.
    result = analyze_quantile_sensitivity(frame, models=["other"], leads=[2], levels=[.2, .8])
    assert result["summary"].model.eq("other").all()
    assert result["summary"].lead.eq(2).all()
    np.testing.assert_allclose(result["rows"].probability_lo, .2)
    np.testing.assert_allclose(result["rows"].probability_hi, .8)
    with pytest.raises(ValueError, match="missing quantile"):
        analyze_quantile_sensitivity(frame)


@pytest.mark.parametrize("change", ["target", "quantile", "probability"])
def test_invalid_numeric_dtypes(change):
    frame = predictions(p=[.4, .975, .025, np.nan])
    column = {"target": "y_L1", "quantile": "smokecast_q0_L1", "probability": "smokecast_prob_L1"}[change]
    frame[column] = frame[column].astype(str)
    with pytest.raises(ValueError, match="real numeric"):
        analyze_quantile_sensitivity(frame)


@pytest.mark.parametrize("kwargs", [{"cutoffs": []}, {"cutoffs": [True]}, {"cutoffs": [np.inf]},
                                    {"cutoffs": [1.1]}, {"cutoffs": [.2, .2]}, {"groups": ["year"]},
                                    {"models": []}, {"leads": [True]}, {"groups": ["point", "point"]}])
def test_invalid_settings(kwargs):
    with pytest.raises((ValueError, TypeError)):
        analyze_quantile_sensitivity(predictions(), **kwargs)


def test_development_dates_and_unique_keys_are_required():
    frame = predictions()
    with pytest.raises(ValueError, match="unique"):
        analyze_quantile_sensitivity(pd.concat([frame, frame.iloc[:1]]))
    with pytest.raises(ValueError, match="nonempty development"):
        analyze_quantile_sensitivity(frame.iloc[:0])
    frame["issue_date"] = pd.date_range("2024-01-01", periods=4)
    with pytest.raises(ValueError, match="development issue"):
        analyze_quantile_sensitivity(frame)
    frame["issue_date"] = pd.date_range("2023-12-28", periods=4)
    with pytest.raises(ValueError, match="reserved target"):
        analyze_quantile_sensitivity(frame)


def test_writer_round_trip_hash_and_no_overwrite(tmp_path):
    frame = predictions(p=[.4, .975, .025, np.nan])
    source, output = tmp_path / "input.parquet", tmp_path / "report"
    frame.to_parquet(source)
    payload = source.read_bytes()
    manifest = write_quantile_sensitivity(source, output)
    assert source.read_bytes() == payload
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["levels"] == list(DEFAULT_LEVELS)
    assert manifest["units"] == "micrograms/m3"
    assert json.loads((output / "manifest.json").read_text()) == manifest
    expected = analyze_quantile_sensitivity(frame)
    pd.testing.assert_frame_equal(pd.read_parquet(output / "rows.parquet"), expected["rows"])
    pd.testing.assert_frame_equal(pd.read_csv(output / "summary.csv"), expected["summary"], check_dtype=False)
    with pytest.raises(FileExistsError):
        write_quantile_sensitivity(source, output)


def test_cli_custom_levels_and_final_lock(tmp_path, capsys):
    source, output = tmp_path / "input.parquet", tmp_path / "report"
    predictions().to_parquet(source)
    assert main(["--predictions", str(source), "--out", str(output), "--model", "smokecast", "--lead", "1",
                 "--level", "0.1", "--level", "0.2", "--level", "0.5", "--level", "0.8", "--level", "0.9",
                 "--cutoff", "0.3", "--group", "point"]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cutoffs"] == [.3]
    assert manifest["levels"] == [.1, .2, .5, .8, .9]
    assert manifest["groups"] == ["point"]
    (tmp_path / "FINAL_RUN_LOCK.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        main(["--predictions", str(source), "--out", str(tmp_path / "reserved")])
    assert exc.value.code == 2
    assert "development inputs" in capsys.readouterr().err
    assert not (tmp_path / "reserved").exists()
