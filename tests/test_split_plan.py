from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from serein.eval.split_plan import main, plan_splits, read_plan, validate_plan, write_plan


def table(days, points=("001", "002"), leads=(0, 1, 2, 3)):
    rows = []
    for day in pd.to_datetime(days):
        for point in points:
            row = {"issue_date": day, "point_id": point, "kind": "epa"}
            for lead in leads:
                row[f"y_L{lead}"] = float((day + pd.Timedelta(days=lead)).day_of_year)
            rows.append(row)
    return pd.DataFrame(rows)


def plan():
    return {role: [{"start": start, "end": end}] for role, start, end in [
        ("train", "2021-06-01", "2021-06-02"),
        ("tune", "2023-05-01", "2023-05-03"),
        ("calibrate", "2023-05-10", "2023-05-12"),
        ("score", "2023-05-20", "2023-05-22"),
    ]}


def inputs():
    return {"train": table(["2021-06-01", "2021-06-02"]),
            "calibrate": table(pd.date_range("2023-05-01", "2023-05-22"))}


def test_disjoint_plan_reports_roles_availability_and_unassigned_rows():
    tables, spec = inputs(), plan()
    before = {name: frame.copy(deep=True) for name, frame in tables.items()}
    original_plan = copy.deepcopy(spec)
    result = plan_splits(tables, spec)
    report = result["report"]
    assert report["calendar_separated"]
    assert report["forward_ordered"]
    assert report["all_roles_have_rows"]
    assert report["all_role_leads_have_targets"]
    assert report["n_shared_labels"] == 0
    assert report["n_input_epa_rows"] == 48
    assert report["n_assigned_rows"] == 22
    assert report["n_unassigned_rows"] == 26
    assert result["roles"].set_index("role").n_rows.to_dict() == {
        "train": 4, "tune": 6, "calibrate": 6, "score": 6}
    assert len(result["availability"]) == 16
    assert len(result["point_availability"]) == 32
    assert result["assignments"].point_id.tolist().count("001") == 11
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], before[name])
    assert spec == original_plan


def test_cross_lead_overlap_is_reported_without_automatically_purging():
    spec = plan()
    spec["score"] = [{"start": "2023-05-13", "end": "2023-05-15"}]
    result = plan_splits(inputs(), spec)
    assert not result["report"]["calendar_separated"]
    assert result["report"]["n_shared_labels"] == 6
    conflict = result["calendar_conflicts"].query("role_a == 'calibrate' and role_b == 'score'")
    assert conflict.n_conflicting_issue_days_a.tolist() == [3]
    assert conflict.n_conflicting_issue_days_b.tolist() == [3]
    assert result["roles"].set_index("role").loc["calibrate", "n_rows"] == 6


def test_missing_observations_and_sites_do_not_relax_calendar_guard():
    spec = plan()
    spec["score"] = [{"start": "2023-05-13", "end": "2023-05-15"}]
    data = inputs()
    data["calibrate"] = data["calibrate"].loc[lambda x: x.issue_date.ne("2023-05-13")].copy()
    data["calibrate"].loc[:, [f"y_L{x}" for x in range(4)]] = np.nan
    result = plan_splits(data, spec)
    assert not result["report"]["calendar_separated"]
    assert result["report"]["n_shared_labels"] == 0
    assert not result["report"]["all_role_leads_have_targets"]
    assert result["ranges"].set_index("role").loc["score", "n_absent_issue_days"] == 1


@pytest.mark.parametrize("leads", [(0,), (0, 1, 2, 3), (1, 5, 9)])
@pytest.mark.parametrize("gap", [0, 1, 4])
def test_calendar_conflicts_match_independent_set_oracle(leads, gap):
    spec = plan()
    data = {"observations": table(pd.date_range("2023-05-01", "2023-05-22"), leads=leads)}
    result = plan_splits(data, spec, leads=leads, gap_days=gap)
    schedules = {
        role: [date.fromisoformat(bounds["start"]) + timedelta(days=i)
               for bounds in ranges
               for i in range((date.fromisoformat(bounds["end"]) -
                               date.fromisoformat(bounds["start"])).days + 1)]
        for role, ranges in spec.items()
    }
    for pair in result["calendar_conflicts"].itertuples():
        for a, b, actual in [(pair.role_a, pair.role_b, pair.n_conflicting_issue_days_a),
                             (pair.role_b, pair.role_a, pair.n_conflicting_issue_days_b)]:
            targets = {d + timedelta(days=lead) for d in schedules[b] for lead in leads}
            expected = sum(any(abs((d + timedelta(days=lead) - t).days) <= gap
                               for lead in leads for t in targets) for d in schedules[a])
            assert actual == expected


def test_multiple_ranges_unsorted_and_nondefault_indexes():
    spec = plan()
    spec["tune"] = [{"start": "2023-05-07", "end": "2023-05-07"}, *spec["tune"]]
    data = inputs()
    data["calibrate"].index = ["duplicate"] * len(data["calibrate"])
    result = plan_splits(data, spec, leads=[0])
    assert result["roles"].set_index("role").loc["tune", "n_rows"] == 8
    assert len(result["ranges"]) == 5
    assert result["ranges"].query("role == 'tune'").start.tolist() == [
        pd.Timestamp("2023-05-01"), pd.Timestamp("2023-05-07")]


def test_empty_role_and_all_non_epa_inputs_are_visible():
    data = inputs()
    for frame in data.values():
        frame["kind"] = "purpleair"
    result = plan_splits(data, plan())
    assert result["report"]["excluded_non_epa_rows"] == {"train": 4, "calibrate": 44}
    assert not result["report"]["all_roles_have_rows"]
    assert not result["report"]["all_role_leads_have_targets"]
    assert result["report"]["calendar_separated"]
    assert result["assignments"].empty
    assert result["point_availability"].columns.tolist() == [
        "role", "lead", "point_id", "n_rows", "n_finite_targets"]
    assert result["ranges"].n_absent_issue_days.tolist() == [2, 3, 3, 3]


def test_nullable_kind_missing_rows_are_counted_as_excluded():
    data = inputs()
    data["train"]["kind"] = pd.array(["epa", None, "purpleair", "epa"], dtype="string")
    result = plan_splits(data, plan())
    assert result["report"]["excluded_non_epa_rows"]["train"] == 2
    assert result["roles"].set_index("role").loc["train", "n_rows"] == 2


def test_separated_calendars_can_have_reversed_role_order():
    spec = plan()
    spec["tune"], spec["score"] = spec["score"], spec["tune"]
    result = plan_splits(inputs(), spec)
    assert result["report"]["calendar_separated"]
    assert not result["report"]["forward_ordered"]
    row = result["role_order"].query("role_a == 'tune' and role_b == 'score'").iloc[0]
    assert row.last_target_a == pd.Timestamp("2023-05-25")
    assert row.first_target_b == pd.Timestamp("2023-05-01")
    assert not row.a_strictly_precedes_b


def test_point_support_distinguishes_missing_sites_and_missing_targets():
    data = inputs()
    frame = data["calibrate"]
    frame.loc[frame.issue_date.between("2023-05-10", "2023-05-12") &
              frame.point_id.eq("002"), "y_L1"] = np.nan
    result = plan_splits(data, plan())
    pair = result["point_support"].query(
        "role_a == 'calibrate' and role_b == 'score' and lead == 1").iloc[0]
    assert pair.n_points_a == 1
    assert pair.n_points_b == 2
    assert pair.n_points_shared == 1
    assert pair.n_points_only_b == 1
    row = result["point_availability"].query(
        "role == 'calibrate' and point_id == '002' and lead == 1").iloc[0]
    assert row.n_rows == 3
    assert row.n_finite_targets == 0


def test_custom_tables_without_kind_and_sparse_leads():
    data = {"archive": table(["2023-05-01", "2023-05-10"], leads=[1, 5]).drop(columns="kind")}
    result = plan_splits(data, plan(), leads=[1, 5])
    assert result["report"]["excluded_non_epa_rows"] == {"archive": 0}
    assert result["availability"].lead.tolist() == [1, 5] * 4
    assert set(result["assignments"].source_table) == {"archive"}


@pytest.mark.parametrize("bad", [None, [], {}, {"train": []}, {**plan(), "extra": []}])
def test_bad_plan_roles(bad):
    with pytest.raises(ValueError, match="exactly these roles"):
        validate_plan(bad)


@pytest.mark.parametrize("bad", [None, [], "2023-05-01", [{}], [["2023-05-01", "2023-05-02"]],
                                     [{"start": "2023-05-01", "end": "2023-05-02", "typo": 1}]])
def test_bad_range_structure(bad):
    spec = plan()
    spec["score"] = bad
    with pytest.raises(ValueError, match="range"):
        validate_plan(spec)


@pytest.mark.parametrize("bad", [None, 20230501, True, "bad", "2023-5-01", "2023-02-29",
                                     "2023-05-01T00:00", "2023-05-01Z", "2024-01-01"])
def test_bad_range_dates(bad):
    spec = plan()
    spec["score"][0]["start"] = bad
    with pytest.raises(ValueError):
        validate_plan(spec)


def test_reversed_and_overlapping_ranges():
    spec = plan()
    spec["score"][0]["end"] = "2023-05-19"
    with pytest.raises(ValueError, match="must not follow"):
        validate_plan(spec)
    spec = plan()
    spec["score"] *= 2
    with pytest.raises(ValueError, match="within score"):
        validate_plan(spec)
    spec = plan()
    spec["score"] = [{"start": "2023-05-12", "end": "2023-05-20"}]
    with pytest.raises(ValueError, match="between calibrate and score"):
        validate_plan(spec)


@pytest.mark.parametrize("lead", [0, 1, 3])
def test_reserved_issue_years_rejected_even_for_unused_non_epa_rows(lead):
    data = inputs()
    other = table(["2024-06-01"])
    other["kind"] = "purpleair"
    data["train"] = pd.concat([data["train"], other])
    with pytest.raises(ValueError, match="reserved issue years"):
        plan_splits(data, plan(), leads=[lead])


def test_reserved_targets_rejected_in_inputs_or_declared_ranges():
    data = inputs()
    data["train"] = pd.concat([data["train"], table(["2023-12-30"])])
    with pytest.raises(ValueError, match="target dates"):
        plan_splits(data, plan())
    spec = plan()
    spec["score"] = [{"start": "2023-12-29", "end": "2023-12-31"}]
    with pytest.raises(ValueError, match="declared targets"):
        plan_splits(inputs(), spec)
    assert plan_splits(inputs(), spec, leads=[0])["report"]["calendar_separated"]


@pytest.mark.parametrize("column,value,match", [
    ("y_L0", np.inf, "finite"), ("point_id", "", "point_id"),
    ("issue_date", "2023-05-01T01:00", "dates"),
    ("source_table", "x", "reserved"), ("role", "x", "reserved"),
])
def test_bad_input_rows_rejected_even_outside_plan(column, value, match):
    data = inputs()
    data["calibrate"][column] = data["calibrate"][column].astype(object) if column in data[
        "calibrate"] else "x"
    data["calibrate"].loc[10, column] = value
    with pytest.raises(ValueError, match=match):
        plan_splits(data, plan())


def test_duplicates_between_sources_and_conflicting_targets():
    data = inputs()
    data["extra"] = data["train"].iloc[:1].copy()
    with pytest.raises(ValueError, match="across input tables"):
        plan_splits(data, plan())
    data = inputs()
    data["train"].loc[0, "y_L1"] = 1.
    with pytest.raises(ValueError, match="conflicting observations"):
        plan_splits(data, plan())


@pytest.mark.parametrize("leads,gap", [([], 0), ([True], 0), ([1, 1], 0), ([-1], 0),
                                         ([0], -1), ([0], True), ([0], 1.5)])
def test_invalid_settings(leads, gap):
    with pytest.raises((ValueError, TypeError)):
        plan_splits(inputs(), plan(), leads=leads, gap_days=gap)


def test_plan_json_rejects_duplicate_fields(tmp_path):
    source = tmp_path / "plan.json"
    source.write_text('{"train": [], "train": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate plan field"):
        read_plan(source)


def saved_inputs(tmp_path):
    paths = {}
    for name, frame in inputs().items():
        paths[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(paths[name], index=False)
    paths["plan"] = tmp_path / "schedule.json"
    paths["plan"].write_text(json.dumps(plan()), encoding="utf-8")
    return paths


def test_writer_preserves_inputs_and_exports_reproducible_tables(tmp_path):
    paths = saved_inputs(tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items()}
    output = tmp_path / "report"
    report = write_plan(paths["train"], paths["calibrate"], paths["plan"], output)
    assert json.loads((output / "report.json").read_text()) == report
    assert json.loads((output / "plan.json").read_text()) == plan()
    assert len(list(output.glob("*.csv"))) == 10
    assignments = pd.read_csv(output / "assignments.csv", dtype={"point_id": str})
    assert len(assignments) == 22
    assert set(assignments.point_id) == {"001", "002"}
    for name, payload in before.items():
        assert paths[name].read_bytes() == payload
        assert report["sources"][name]["sha256"] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError):
        write_plan(paths["train"], paths["calibrate"], paths["plan"], output)


def test_invalid_plan_does_not_create_output(tmp_path):
    paths = saved_inputs(tmp_path)
    paths["plan"].write_text("{}", encoding="utf-8")
    output = tmp_path / "report"
    with pytest.raises(ValueError):
        write_plan(paths["train"], paths["calibrate"], paths["plan"], output)
    assert not output.exists()


def test_cli_success_and_actionable_errors(tmp_path, capsys):
    paths = saved_inputs(tmp_path)
    args = ["--train-table", str(paths["train"]), "--calibrate-table", str(paths["calibrate"]),
            "--plan", str(paths["plan"]), "--out", str(tmp_path / "result")]
    main(args)
    assert json.loads(capsys.readouterr().out)["calendar_separated"]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "output already exists" in capsys.readouterr().err
