"""Audit reused target labels across development roles without fitting a model."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from serein.eval.comparison import _calendar_dates, _nonnegative_integer
from serein.eval.episode_inputs import _choices, _load_predictions, _values

KEY = ["point_id", "target_date"]
OVERLAP_COLUMNS = ["role_a", "role_b", "n_shared_labels", "n_shared_target_days",
                   "n_exposures_a", "n_exposures_b"]
DAY_COLUMNS = ["role_a", "role_b", "target_date", "n_shared_labels"]


def _table(frame: pd.DataFrame, leads: list[int]) -> pd.DatetimeIndex:
    if not frame.columns.is_unique:
        raise ValueError("table columns must be unique")
    required = ["issue_date", "point_id"] + [f"y_L{lead}" for lead in leads]
    if missing := sorted(set(required) - set(frame)):
        raise ValueError(f"missing table columns: {', '.join(missing)}")
    days = _calendar_dates(frame.issue_date)
    if not frame.point_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("point_id must contain nonempty strings")
    if pd.DataFrame({"point_id": frame.point_id.to_numpy(), "issue_date": days}).duplicated().any():
        raise ValueError("duplicate point/issue-date rows")
    return days


def audit_splits(
    roles: dict[str, pd.DataFrame], *, leads=(0, 1, 2, 3),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count finite (point, local target date) labels shared between roles.

    Repeated labels within one role count once in n_labels, while exposures count
    every issue/lead use. Cross-lead overlap matters even when issue dates differ.
    No features, weights, model checkpoints or observation availability are checked.
    """
    names = _choices(roles, "roles")
    leads = _choices(leads, "leads", integers=True)
    if len(names) < 2:
        raise ValueError("at least two roles are required")
    summaries, labels = [], {}
    all_labels = []
    for name in names:
        frame = roles[name]
        days = _table(frame, leads)
        parts = []
        for lead in leads:
            obs = _values(frame, f"y_L{lead}")
            finite = np.isfinite(obs)
            parts.append(pd.DataFrame({
                "point_id": frame.point_id.to_numpy()[finite],
                "target_date": (days + pd.Timedelta(days=lead))[finite],
                "observed": obs[finite],
            }))
        raw = pd.concat(parts, ignore_index=True)
        all_labels.append(raw)
        labels[name] = raw.groupby(KEY, sort=True).size().rename("n_exposures").reset_index()
        summaries.append({
            "role": name, "n_rows": len(frame), "n_issue_days": days.nunique(),
            "n_target_slots": len(frame) * len(leads), "n_finite_targets": len(raw),
            "n_negative_targets": int(raw.observed.lt(0).sum()),
            "n_missing_targets": len(frame) * len(leads) - len(raw),
            "n_labels": len(labels[name]),
            "n_target_days": labels[name].target_date.nunique(),
        })
    bounds = pd.concat(all_labels, ignore_index=True).groupby(KEY)["observed"].agg(["min", "max"])
    if not np.isclose(bounds["min"], bounds["max"], rtol=1e-6, atol=1e-6).all():
        raise ValueError("conflicting observations for the same point and target date")
    overlaps, shared_days = [], []
    for a, b in itertools.combinations(names, 2):
        common = labels[a].merge(labels[b], on=KEY, suffixes=("_a", "_b"), validate="one_to_one")
        overlaps.append({
            "role_a": a, "role_b": b, "n_shared_labels": len(common),
            "n_shared_target_days": common.target_date.nunique(),
            "n_exposures_a": int(common.n_exposures_a.sum()),
            "n_exposures_b": int(common.n_exposures_b.sum()),
        })
        daily = common.groupby("target_date", sort=True).size().rename("n_shared_labels")
        shared_days.extend({"role_a": a, "role_b": b, "target_date": day,
                            "n_shared_labels": int(count)} for day, count in daily.items())
    return (pd.DataFrame(summaries), pd.DataFrame(overlaps, columns=OVERLAP_COLUMNS),
            pd.DataFrame(shared_days, columns=DAY_COLUMNS))


def purge_issue_days(candidates, reference, *, leads=(0, 1, 2, 3), gap_days=0) -> np.ndarray:
    """Keep whole candidate issue days whose target dates avoid reference targets.

    The calendar rule ignores sites and missing labels. A gap of g requires all
    candidate target days to be more than g calendar days from reference targets.
    Returns a positional mask; neither input nor the reference schedule is changed.
    """
    leads = _choices(leads, "leads", integers=True)
    _nonnegative_integer(gap_days, "gap_days")
    candidate_days, reference_days = _calendar_dates(candidates), _calendar_dates(reference)
    # Normalize resolution explicitly: pandas 2 and 3 can store different time units.
    origin = pd.Timestamp("1970-01-01")
    c = np.asarray((candidate_days - origin).days, dtype=np.int64)
    r = np.asarray((reference_days - origin).days, dtype=np.int64)
    targets = np.unique(np.concatenate([r + lead for lead in leads]))
    keep = np.ones(len(c), dtype=bool)
    if not len(targets):
        return keep
    for lead in leads:
        values = c + lead
        positions = np.searchsorted(targets, values)
        left = targets[np.maximum(positions - 1, 0)]
        right = targets[np.minimum(positions, len(targets) - 1)]
        distance = np.minimum(np.abs(values - left), np.abs(values - right))
        keep &= distance > gap_days
    return keep


def development_roles(train: pd.DataFrame, calibrate: pd.DataFrame, *, leads=(0, 1, 2, 3)):
    """Mirror development mode only: even weeks validate/calibrate, odd weeks score.

    Refuse reserved issue years even when a caller misnames a test table. Only
    issue years are checked; this is not a claim about model or cache provenance.
    """
    leads = _choices(leads, "leads", integers=True)
    for frame in (train, calibrate):
        days = _table(frame, leads)
        if (days.year >= 2024).any():
            raise ValueError("development audit refuses reserved issue years 2024 and later")
        if frame.empty:
            raise ValueError("development input tables must not be empty")
    # The backtest loads only EPA rows. Plain custom tables can omit kind.
    train, calibrate = [frame.loc[frame.kind.eq("epa")] if "kind" in frame else frame
                        for frame in (train, calibrate)]
    if train.empty or calibrate.empty:
        raise ValueError("development roles must contain EPA rows when kind is supplied")
    days = _calendar_dates(calibrate.issue_date)
    even = days.isocalendar().week.to_numpy(dtype=int) % 2 == 0
    if even.all() or not even.any():
        raise ValueError("calibration table must contain both even and odd ISO weeks")
    return {"train": train, "validation": calibrate.loc[even], "score": calibrate.loc[~even]}


def write_development_audit(train_path: Path, calibrate_path: Path, output: Path, *,
                            leads=(0, 1, 2, 3), gap_days=0, plot: bool = False) -> dict:
    """Write original and proposed development audits to a new directory."""
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    leads = [int(lead) for lead in _choices(leads, "leads", integers=True)]
    _nonnegative_integer(gap_days, "gap_days")
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be boolean")
    train, train_hash = _load_predictions(train_path)
    calibrate, calibrate_hash = _load_predictions(calibrate_path)
    roles = development_roles(train, calibrate, leads=leads)
    original = audit_splits(roles, leads=leads)
    val = roles["validation"]
    keep = purge_issue_days(val.issue_date, roles["score"].issue_date,
                            leads=leads, gap_days=gap_days)
    proposed = {**roles, "validation": val.loc[keep]}
    purged = audit_splits(proposed, leads=leads)
    assignments = []
    for name, frame in roles.items():
        days = _calendar_dates(frame.issue_date)
        included = keep if name == "validation" else np.ones(len(frame), dtype=bool)
        assignments.append(pd.DataFrame({"role": name, "issue_date": days,
                                         "keep_proposed": included}).drop_duplicates())
    assignments = pd.concat(assignments, ignore_index=True).sort_values(["role", "issue_date"])
    report = {
        "mode": "development_only", "leads": leads, "gap_days": int(gap_days),
        "sources": {"train": {"path": str(train_path.resolve()), "sha256": train_hash},
                    "calibrate": {"path": str(calibrate_path.resolve()), "sha256": calibrate_hash}},
        "input_rows": {"train": len(train), "calibrate": len(calibrate)},
        "excluded_non_epa_rows": {
            "train": len(train) - len(roles["train"]),
            "calibrate": len(calibrate) - len(roles["validation"]) - len(roles["score"]),
        },
        "original_shared_labels": int(original[1].n_shared_labels.sum()),
        "proposed_shared_labels": int(purged[1].n_shared_labels.sum()),
        "validation_rows_removed": int((~keep).sum()),
        "validation_rows_retained": int(keep.sum()),
        "proposal_has_validation_rows": bool(keep.any()),
        "limitations": [
            "Configured split years and season membership are not checked.",
            "Validation also fits conformal intervals and warning cutoffs in development mode.",
            "Pairwise overlap counts can count a label more than once across three roles.",
            "The proposal removes validation issue days only; training overlap is not repaired.",
            "No model was fitted or scored, and no feature or checkpoint provenance was checked.",
            "Changing the split requires team review and a new development run.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for prefix, tables in (("", original), ("proposed_", purged)):
        for name, table in zip(("roles", "overlaps", "overlap_days"), tables):
            table.to_csv(output / f"{prefix}{name}.csv", index=False)
    assignments.to_csv(output / "issue_assignments.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if plot:
        from serein.eval.split_plots import plot_split_audit

        plot_split_audit(assignments, original[2], purged[2], output / "split_audit.png")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", type=Path, required=True)
    parser.add_argument("--calibrate-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--leads", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--gap-days", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="also save a local calendar figure")
    args = parser.parse_args(argv)
    try:
        report = write_development_audit(args.train_table, args.calibrate_table, args.out,
                                          leads=args.leads, gap_days=args.gap_days, plot=args.plot)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"split audit: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
