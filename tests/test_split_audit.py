from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from serein.eval.split_audit import (
    audit_splits,
    development_roles,
    main,
    purge_issue_days,
    write_development_audit,
)


def table(days, points=("a",), leads=(0, 1, 2, 3)):
    rows = []
    for day in pd.to_datetime(days):
        for point in points:
            rows.append({"issue_date": day, "point_id": point, **{
                f"y_L{lead}": float((day + pd.Timedelta(days=lead)).day_of_year)
                for lead in leads}})
    return pd.DataFrame(rows, columns=["issue_date", "point_id"] + [f"y_L{x}" for x in leads])


def test_cross_lead_overlap_counts_labels_and_exposures():
    a = table(["2023-06-04", "2023-06-05"])
    b = table(["2023-06-07"])
    roles, pairs, daily = audit_splits({"validation": a, "score": b})
    assert roles.set_index("role").loc["validation"].to_dict() == {
        "n_rows": 2, "n_issue_days": 2, "n_target_slots": 8,
        "n_finite_targets": 8, "n_missing_targets": 0, "n_labels": 5, "n_target_days": 5,
        "n_negative_targets": 0,
    }
    assert pairs.iloc[0].to_dict() == {
        "role_a": "validation", "role_b": "score", "n_shared_labels": 2,
        "n_shared_target_days": 2, "n_exposures_a": 3, "n_exposures_b": 2,
    }
    assert daily.target_date.tolist() == [pd.Timestamp("2023-06-07"), pd.Timestamp("2023-06-08")]
    assert daily.n_shared_labels.tolist() == [1, 1]
    assert audit_splits({"validation": a, "score": b}, leads=[0])[1].n_shared_labels.sum() == 0


def test_overlap_needs_same_point_and_finite_label():
    a = table(["2023-06-04"], points=("a", "b"))
    b = table(["2023-06-05"], points=("b", "c"))
    b.loc[b.point_id.eq("b"), "y_L1"] = np.nan
    roles, pairs, daily = audit_splits({"train": a, "score": b})
    assert pairs.n_shared_labels.tolist() == [2]
    assert roles.n_missing_targets.tolist() == [0, 1]
    assert daily.target_date.tolist() == [pd.Timestamp("2023-06-05"), pd.Timestamp("2023-06-07")]


def test_every_pair_reported_including_disjoint_and_empty():
    a = table(["2021-06-01"])
    b = table(["2023-06-01"])
    empty = b.iloc[:0]
    roles, pairs, days = audit_splits({"train": a, "validation": b, "score": empty})
    assert pairs.n_shared_labels.tolist() == [0, 0, 0]
    assert roles.n_rows.tolist() == [1, 1, 0]
    assert days.empty
    assert days.columns.tolist() == ["role_a", "role_b", "target_date", "n_shared_labels"]


def test_all_missing_roles_and_numpy_nullable_data():
    a = table(["2023-06-01"])
    for lead in range(4):
        a[f"y_L{lead}"] = pd.array([None], dtype="Float64")
    roles, pairs, _ = audit_splits({"a": a, "b": a.copy()})
    assert roles.n_finite_targets.tolist() == [0, 0]
    assert pairs.n_shared_labels.tolist() == [0]


@pytest.mark.parametrize("change", ["same_role", "other_role"])
def test_conflicting_targets_rejected(change):
    a = table(["2023-06-01", "2023-06-02"])
    b = table(["2023-06-02"])
    if change == "same_role":
        a.loc[0, "y_L1"] = 1.
    else:
        b.loc[0, "y_L0"] = 1.
    with pytest.raises(ValueError, match="conflicting observations"):
        audit_splits({"a": a, "b": b})


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_invalid_concentrations(value):
    a = table(["2023-06-01"])
    a.loc[0, "y_L0"] = value
    with pytest.raises(ValueError, match="finite"):
        audit_splits({"a": a, "b": a.copy()})


def test_negative_observations_remain_labels_and_are_counted():
    a = table(["2023-06-01"])
    a["y_L0"] = -2.
    roles, pairs, _ = audit_splits({"a": a, "b": a.copy()})
    assert roles.n_negative_targets.tolist() == [1, 1]
    assert pairs.n_shared_labels.tolist() == [4]


@pytest.mark.parametrize("point", [None, "", " ", 1, True])
def test_point_identifiers(point):
    a = table(["2023-06-01"])
    a["point_id"] = [point]
    with pytest.raises(ValueError, match="point_id"):
        audit_splits({"a": a, "b": a.copy()})


@pytest.mark.parametrize("day", [None, "invalid", "2023-06-01T01:00", "2023-06-01T00:00Z", 12])
def test_dates_rejected(day):
    a = table(["2023-06-01"])
    a["issue_date"] = [day]
    with pytest.raises(ValueError, match="dates"):
        audit_splits({"a": a, "b": a.copy()})


def test_duplicates_checked_after_date_normalization():
    a = table(["2023-06-01", "2023-06-02"])
    a["issue_date"] = [date(2023, 6, 1), "2023-06-01"]
    with pytest.raises(ValueError, match="duplicate"):
        audit_splits({"a": a, "b": a.copy()})
    a = table(["2023-06-01"])
    a = pd.concat([a, a[["y_L0"]]], axis=1)
    with pytest.raises(ValueError, match="columns must be unique"):
        audit_splits({"a": a, "b": a.copy()})


@pytest.mark.parametrize("leads", [[], [1, 1], [True], [-1], [1.5]])
def test_bad_leads(leads):
    a = table(["2023-06-01"])
    with pytest.raises(ValueError, match="leads"):
        audit_splits({"a": a, "b": a}, leads=leads)


def test_missing_columns_and_roles():
    a = table(["2023-06-01"])
    with pytest.raises(ValueError, match="missing table columns: y_L1"):
        audit_splits({"a": a.drop(columns="y_L1"), "b": a})
    with pytest.raises(ValueError, match="at least two"):
        audit_splits({"a": a})
    with pytest.raises(ValueError, match="roles"):
        audit_splits({"not a role": a, "b": a})


@pytest.mark.parametrize("gap", [0, 1, 3, 10])
@pytest.mark.parametrize("leads", [(0,), (0, 1, 2, 3), (1, 4, 8)])
def test_purge_matches_independent_calendar_oracle(gap, leads):
    start = date(2023, 6, 1)
    candidates = [start + timedelta(days=i) for i in range(40)]
    reference = [start + timedelta(days=i) for i in [6, 7, 20, 24]]
    ref_targets = {day + timedelta(days=lead) for day in reference for lead in leads}
    expected = [all(abs((day + timedelta(days=lead) - target).days) > gap
                    for lead in leads for target in ref_targets) for day in candidates]
    actual = purge_issue_days(candidates, reference, leads=leads, gap_days=gap)
    np.testing.assert_array_equal(actual, expected)


def test_purge_handles_empty_schedules_duplicates_and_calendar_gaps():
    days = ["2023-01-01", "2023-01-01", "2023-01-10"]
    np.testing.assert_array_equal(purge_issue_days(days, []), [True] * 3)
    np.testing.assert_array_equal(purge_issue_days([], days), [])
    np.testing.assert_array_equal(purge_issue_days(days, ["2023-01-05"]), [True, True, True])
    np.testing.assert_array_equal(purge_issue_days(days, ["2023-01-04"]), [False, False, True])


@pytest.mark.parametrize("gap", [-1, True, 1.5, np.nan])
def test_invalid_gap(gap):
    with pytest.raises(ValueError, match="gap_days"):
        purge_issue_days([], [], gap_days=gap)


def test_development_roles_match_week_parity_and_keep_indices():
    train = table(["2021-06-01"])
    calibrate = table(pd.date_range("2023-06-01", periods=30))
    calibrate.index = np.arange(100, 130)
    original = calibrate.copy(deep=True)
    roles = development_roles(train, calibrate)
    for role, parity in [("validation", 0), ("score", 1)]:
        assert all(day.isocalendar().week % 2 == parity for day in roles[role].issue_date)
    assert sum(len(frame) for frame in roles.values()) == 31
    pd.testing.assert_frame_equal(original, calibrate)


@pytest.mark.parametrize("which", ["train", "calibrate"])
@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_reserved_years_refused(which, year):
    frames = {"train": table(["2021-06-01"]),
              "calibrate": table(["2023-06-01", "2023-06-08"])}
    frames[which] = table([f"{year}-06-01"])
    with pytest.raises(ValueError, match="reserved"):
        development_roles(**frames)


def test_development_rejects_empty_or_one_parity():
    train = table(["2021-06-01"])
    with pytest.raises(ValueError, match="must not be empty"):
        development_roles(train.iloc[:0], train)
    with pytest.raises(ValueError, match="both even and odd"):
        development_roles(train, table(["2023-06-01"]))


def test_non_epa_rows_do_not_enter_development_roles(tmp_path):
    train, calibrate = save_tables(tmp_path)
    for path in (train, calibrate):
        frame = pd.read_parquet(path).assign(kind="epa")
        extra = frame.assign(point_id="low_cost_" + frame.point_id, kind="purpleair")
        pd.concat([frame, extra]).to_parquet(path, index=False)
    report = write_development_audit(train, calibrate, tmp_path / "epa")
    assert report["excluded_non_epa_rows"] == {"train": 1, "calibrate": 60}
    assert report["input_rows"] == {"train": 2, "calibrate": 120}
    roles = development_roles(pd.read_parquet(train), pd.read_parquet(calibrate))
    assert all(frame.kind.eq("epa").all() for frame in roles.values())


def test_all_non_epa_rows_are_refused():
    train = table(["2021-06-01"]).assign(kind="purpleair")
    calibrate = table(["2023-06-01", "2023-06-08"]).assign(kind="epa")
    with pytest.raises(ValueError, match="must contain EPA"):
        development_roles(train, calibrate)


def save_tables(tmp_path):
    train = tmp_path / "train.parquet"
    calibrate = tmp_path / "calibrate.parquet"
    table(["2021-06-01"]).to_parquet(train)
    table(pd.date_range("2023-06-01", periods=30), points=("a", "b")).to_parquet(calibrate)
    return train, calibrate


def test_written_audit_hashes_sources_and_preserves_inputs(tmp_path):
    train, calibrate = save_tables(tmp_path)
    before = {path: path.read_bytes() for path in (train, calibrate)}
    output = tmp_path / "new report"
    report = write_development_audit(train, calibrate, output)
    assert report["original_shared_labels"] > 0
    assert report["proposed_shared_labels"] == 0
    assert report["proposal_has_validation_rows"]
    assert report["validation_rows_retained"] > 0
    assert report["validation_rows_removed"] > 0
    assert json.loads((output / "report.json").read_text()) == report
    for role, path in [("train", train), ("calibrate", calibrate)]:
        assert path.read_bytes() == before[path]
        assert report["sources"][role]["sha256"] == hashlib.sha256(before[path]).hexdigest()
    assignments = pd.read_csv(output / "issue_assignments.csv")
    assert not assignments.duplicated(["role", "issue_date"]).any()
    assert assignments.loc[assignments.role.ne("validation"), "keep_proposed"].all()
    assert {path.name for path in output.iterdir()} == {
        "report.json", "issue_assignments.csv", "roles.csv", "overlaps.csv", "overlap_days.csv",
        "proposed_roles.csv", "proposed_overlaps.csv", "proposed_overlap_days.csv",
    }
    with pytest.raises(FileExistsError, match="already exists"):
        write_development_audit(train, calibrate, output)


def test_proposal_can_be_unusable_and_does_not_repair_train_overlap(tmp_path):
    train, calibrate = save_tables(tmp_path)
    table(["2023-06-08"]).to_parquet(train)
    report = write_development_audit(train, calibrate, tmp_path / "out", gap_days=400)
    assert not report["proposal_has_validation_rows"]
    assert report["validation_rows_retained"] == 0
    assert report["proposed_shared_labels"] > 0


def test_cli_success_and_errors(tmp_path, capsys):
    train, calibrate = save_tables(tmp_path)
    args = ["--train-table", str(train), "--calibrate-table", str(calibrate),
            "--out", str(tmp_path / "cli")]
    main(args)
    assert json.loads(capsys.readouterr().out)["proposed_shared_labels"] == 0
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert "already exists" in capsys.readouterr().err


def test_numpy_integer_settings_serialize(tmp_path):
    train, calibrate = save_tables(tmp_path)
    output = tmp_path / "numpy_settings"
    report = write_development_audit(train, calibrate, output,
                                     leads=np.array([0, 1, 2, 3]), gap_days=np.int64(0))
    assert json.loads((output / "report.json").read_text()) == report


def test_optional_calendar_plot_from_cli(tmp_path, capsys):
    train, calibrate = save_tables(tmp_path)
    output = tmp_path / "with_plot"
    main(["--train-table", str(train), "--calibrate-table", str(calibrate),
          "--out", str(output), "--plot"])
    assert json.loads(capsys.readouterr().out)["proposed_shared_labels"] == 0
    assert (output / "split_audit.png").read_bytes().startswith(b"\x89PNG")


def test_plot_setting_must_be_boolean(tmp_path):
    train, calibrate = save_tables(tmp_path)
    output = tmp_path / "bad_plot"
    with pytest.raises(TypeError, match="plot must be boolean"):
        write_development_audit(train, calibrate, output, plot="yes")
    assert not output.exists()


def test_cli_bad_input_does_not_create_output(tmp_path, capsys):
    train, calibrate = save_tables(tmp_path)
    table(["2025-06-01"]).to_parquet(calibrate)
    output = tmp_path / "refused"
    with pytest.raises(SystemExit) as exc:
        main(["--train-table", str(train), "--calibrate-table", str(calibrate), "--out", str(output)])
    assert exc.value.code == 2
    assert "reserved" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("seed", range(8))
def test_audit_matches_label_set_oracle_under_missingness(seed):
    rng = np.random.default_rng(seed)
    roles = {}
    leads = [0, 2, 5]
    expected = {}
    for name in ["train", "validation", "score"]:
        days = pd.to_datetime("2023-06-01") + pd.to_timedelta(
            rng.choice(20, size=8, replace=False), unit="D")
        frame = table(days, points=("0001", "0002", "third"), leads=leads)
        for lead in leads:
            frame.loc[rng.random(len(frame)) < .3, f"y_L{lead}"] = np.nan
        frame = frame.sample(frac=1, random_state=seed)
        roles[name] = frame
        uses = {}
        for row in frame.to_dict("records"):
            for lead in leads:
                if pd.notna(row[f"y_L{lead}"]):
                    key = row["point_id"], row["issue_date"].date() + timedelta(days=lead)
                    uses[key] = uses.get(key, 0) + 1
        expected[name] = uses
    summaries, overlaps, daily = audit_splits(roles, leads=leads)
    for record in summaries.to_dict("records"):
        uses = expected[record["role"]]
        assert record["n_labels"] == len(uses)
        assert record["n_finite_targets"] == sum(uses.values())
    for record in overlaps.to_dict("records"):
        a, b = record["role_a"], record["role_b"]
        shared = expected[a].keys() & expected[b].keys()
        assert record["n_shared_labels"] == len(shared)
        assert record["n_exposures_a"] == sum(expected[a][key] for key in shared)
        assert record["n_exposures_b"] == sum(expected[b][key] for key in shared)
        subset = daily.loc[daily.role_a.eq(a) & daily.role_b.eq(b)]
        assert int(subset.n_shared_labels.sum()) == len(shared)


def test_datetime_storage_resolution_does_not_change_purge():
    candidates = pd.date_range("2023-06-01", periods=20)
    reference = pd.date_range("2023-06-08", periods=2)
    expected = purge_issue_days(candidates, reference)
    for unit in ["ns", "us", "ms", "s"]:
        np.testing.assert_array_equal(
            purge_issue_days(candidates.as_unit(unit), reference.as_unit(unit)), expected)


def test_validation_purge_is_global_across_points_and_missing_targets(tmp_path):
    train, calibrate = save_tables(tmp_path)
    frame = pd.read_parquet(calibrate)
    frame.loc[frame.point_id.eq("a"), [f"y_L{x}" for x in range(4)]] = np.nan
    frame.to_parquet(calibrate)
    output = tmp_path / "global"
    report = write_development_audit(train, calibrate, output)
    assert report["validation_rows_removed"] == 2 * (
        pd.read_csv(output / "issue_assignments.csv")
        .query("role == 'validation' and not keep_proposed").shape[0])
