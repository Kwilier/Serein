"""Compare explicit development role schedules before fitting new models."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from serein.eval.comparison import _calendar_dates, _nonnegative_integer
from serein.eval.episode_inputs import _choices, _load_predictions, _values
from serein.eval.split_audit import _table, audit_splits, purge_issue_days

ROLES = ("train", "tune", "calibrate", "score")


def _date(value) -> pd.Timestamp:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("schedule dates must use YYYY-MM-DD")
    day = _calendar_dates([value])[0]
    if day.year >= 2024:
        raise ValueError("development plans refuse reserved dates 2024 and later")
    return day


def validate_plan(plan: dict) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Validate four explicit roles with inclusive, disjoint issue-date ranges."""
    if not isinstance(plan, dict) or set(plan) != set(ROLES):
        raise ValueError(f"plan must contain exactly these roles: {', '.join(ROLES)}")
    parsed = {}
    for role in ROLES:
        ranges = plan[role]
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"{role} must contain a nonempty list of ranges")
        parsed[role] = []
        for bounds in ranges:
            if not isinstance(bounds, dict) or set(bounds) != {"start", "end"}:
                raise ValueError("each range must contain only start and end")
            start, end = _date(bounds["start"]), _date(bounds["end"])
            if start > end:
                raise ValueError("range start must not follow end")
            parsed[role].append((start, end))
        parsed[role].sort()
        if any(b[0] <= a[1] for a, b in itertools.pairwise(parsed[role])):
            raise ValueError(f"overlapping issue-date ranges within {role}")
    for a, b in itertools.combinations(ROLES, 2):
        if any(max(x[0], y[0]) <= min(x[1], y[1]) for x in parsed[a] for y in parsed[b]):
            raise ValueError(f"issue-date ranges overlap between {a} and {b}")
    return parsed


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate plan field: {key}")
        result[key] = value
    return result


def read_plan(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    plan = json.loads(payload, object_pairs_hook=_no_duplicate_keys)
    validate_plan(plan)
    return plan, hashlib.sha256(payload).hexdigest()


def plan_splits(tables: dict[str, pd.DataFrame], plan: dict, *,
                leads=(0, 1, 2, 3), gap_days=0) -> dict:
    """Audit a fixed plan without selecting dates from model outcomes.

    Calendar checks use declared ranges even if observations or whole issue days
    are missing. Label audits use only available EPA rows and finite targets.
    Neither kind of check establishes statistical independence or sample adequacy.
    """
    ranges = validate_plan(plan)
    leads = _choices(leads, "leads", integers=True)
    _nonnegative_integer(gap_days, "gap_days")
    sources = _choices(tables, "table names")
    frames, excluded = [], {}
    for name in sources:
        frame = tables[name]
        if "source_table" in frame or "role" in frame:
            raise ValueError("input columns role and source_table are reserved")
        days = _table(frame, leads)
        if (days.year >= 2024).any():
            raise ValueError("development plans refuse reserved issue years 2024 and later")
        if ((days + pd.Timedelta(days=max(leads))).year >= 2024).any():
            raise ValueError("development plans refuse target dates 2024 and later")
        # Validate unused rows too, so a revised plan cannot expose hidden bad values.
        for lead in leads:
            _values(frame, f"y_L{lead}")
        epa = (frame.kind.eq("epa").fillna(False).to_numpy(dtype=bool)
               if "kind" in frame else np.ones(len(frame), dtype=bool))
        excluded[name] = int((~epa).sum())
        frame = frame.loc[epa].copy()
        frame["issue_date"] = days[epa]
        frame["source_table"] = name
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True)
    if pool.duplicated(["point_id", "issue_date"]).any():
        raise ValueError("duplicate point/issue-date rows across input tables")
    roles, schedules, range_rows = {}, {}, []
    for role in ROLES:
        selected = np.zeros(len(pool), dtype=bool)
        declared = []
        for start, end in ranges[role]:
            calendar = pd.date_range(start, end)
            if ((calendar + pd.Timedelta(days=max(leads))).year >= 2024).any():
                raise ValueError("development plans refuse declared targets in 2024 and later")
            mask = pool.issue_date.between(start, end).to_numpy()
            selected |= mask
            declared.append(calendar)
            available = pool.loc[mask]
            n_days = available.issue_date.nunique()
            range_rows.append({
                "role": role, "start": start, "end": end,
                "n_declared_issue_days": len(calendar), "n_available_issue_days": n_days,
                "n_absent_issue_days": len(calendar) - n_days,
                "n_rows": len(available), "n_points": available.point_id.nunique(),
            })
        roles[role] = pool.loc[selected].copy()
        schedules[role] = declared[0].append(declared[1:])
    summary, overlaps, overlap_days = audit_splits(roles, leads=leads)
    calendar_pairs, ordering = [], []
    for a, b in itertools.combinations(ROLES, 2):
        keep_a = purge_issue_days(schedules[a], schedules[b], leads=leads, gap_days=gap_days)
        keep_b = purge_issue_days(schedules[b], schedules[a], leads=leads, gap_days=gap_days)
        calendar_pairs.append({"role_a": a, "role_b": b,
                               "n_conflicting_issue_days_a": int((~keep_a).sum()),
                               "n_conflicting_issue_days_b": int((~keep_b).sum())})
        ordering.append({
            "role_a": a, "role_b": b,
            "last_target_a": schedules[a].max() + pd.Timedelta(days=max(leads)),
            "first_target_b": schedules[b].min() + pd.Timedelta(days=min(leads)),
        })
    ordering = pd.DataFrame(ordering)
    ordering["a_strictly_precedes_b"] = ordering.last_target_a < ordering.first_target_b
    availability, points, support = [], [], {}
    for role, frame in roles.items():
        for lead in leads:
            finite = np.isfinite(_values(frame, f"y_L{lead}"))
            counts = pd.DataFrame({"point_id": frame.point_id.to_numpy(), "finite": finite})
            grouped = counts.groupby("point_id", sort=True).finite.agg(["size", "sum"])
            support[role, lead] = set(grouped.index[grouped["sum"].gt(0)])
            for point, row in grouped.iterrows():
                points.append({"role": role, "lead": lead, "point_id": point,
                               "n_rows": int(row["size"]), "n_finite_targets": int(row["sum"])})
            availability.append({"role": role, "lead": lead, "n_rows": len(frame),
                                 "n_finite_targets": int(finite.sum()),
                                 "n_missing_targets": int((~finite).sum()),
                                 "n_points_with_targets": int((grouped["sum"] > 0).sum())})
    point_pairs = []
    for a, b in itertools.combinations(ROLES, 2):
        for lead in leads:
            sa, sb = support[a, lead], support[b, lead]
            point_pairs.append({
                "role_a": a, "role_b": b, "lead": lead,
                "n_points_a": len(sa), "n_points_b": len(sb),
                "n_points_shared": len(sa & sb),
                "n_points_only_a": len(sa - sb), "n_points_only_b": len(sb - sa),
            })
    assigned = pd.concat([frame.assign(role=role) for role, frame in roles.items()],
                         ignore_index=True)
    assignments = assigned[["role", "source_table", "point_id", "issue_date"]].sort_values(
        ["role", "issue_date", "point_id"])
    calendar_pairs = pd.DataFrame(calendar_pairs)
    availability = pd.DataFrame(availability)
    report = {
        "mode": "development_plan_only", "leads": [int(x) for x in leads],
        "gap_days": int(gap_days), "excluded_non_epa_rows": excluded,
        "n_input_epa_rows": len(pool), "n_assigned_rows": len(assigned),
        "n_unassigned_rows": len(pool) - len(assigned),
        "all_roles_have_rows": bool(summary.n_rows.gt(0).all()),
        "all_role_leads_have_targets": bool(availability.n_finite_targets.gt(0).all()),
        "calendar_separated": bool(calendar_pairs.filter(like="n_conflicting").eq(0).all().all()),
        "forward_ordered": bool(ordering.a_strictly_precedes_b.all()),
        "n_shared_labels": int(overlaps.n_shared_labels.sum()),
        "limitations": [
            "This plan is not applied to training, calibration or scoring.",
            "No model outcomes were used to select or rank schedules.",
            "Nonempty roles and zero shared labels do not establish sample adequacy or independence.",
            "Configured seasons, feature availability and checkpoint provenance are not checked.",
            "Calendar checks include missing days and targets; label counts use available EPA rows.",
            "Calendar separation does not require forward-in-time role ordering.",
            "Team review and a fresh development run are required before adopting any plan.",
        ],
    }
    return {"report": report, "roles": summary, "overlaps": overlaps,
            "overlap_days": overlap_days, "ranges": pd.DataFrame(range_rows),
            "calendar_conflicts": calendar_pairs, "role_order": ordering,
            "point_support": pd.DataFrame(point_pairs), "availability": availability,
            "point_availability": pd.DataFrame(points, columns=[
                "role", "lead", "point_id", "n_rows", "n_finite_targets"]),
            "assignments": assignments}


def write_plan(train_path: Path, calibrate_path: Path, plan_path: Path, output: Path, *,
               leads=(0, 1, 2, 3), gap_days=0, plot: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be boolean")
    plan, plan_hash = read_plan(plan_path)
    train, train_hash = _load_predictions(train_path)
    calibrate, calibrate_hash = _load_predictions(calibrate_path)
    result = plan_splits({"train": train, "calibrate": calibrate}, plan,
                         leads=leads, gap_days=gap_days)
    report = result.pop("report")
    report["sources"] = {
        name: {"path": str(path.resolve()), "sha256": digest}
        for name, path, digest in (("train", train_path, train_hash),
                                   ("calibrate", calibrate_path, calibrate_hash),
                                   ("plan", plan_path, plan_hash))
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in result.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if plot:
        from serein.eval.split_plan_plots import plot_plan

        plot_plan(result["ranges"], result["assignments"], result["availability"],
                  output / "development_plan.png")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", type=Path, required=True)
    parser.add_argument("--calibrate-table", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--leads", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--gap-days", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="save a local calendar and target figure")
    args = parser.parse_args(argv)
    try:
        report = write_plan(args.train_table, args.calibrate_table, args.plan, args.out,
                            leads=args.leads, gap_days=args.gap_days, plot=args.plot)
    except (ValueError, TypeError, OSError, OverflowError) as exc:
        parser.exit(2, f"split plan: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
