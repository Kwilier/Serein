"""Compare development input distributions against a fixed reference table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from serein.eval.comparison import _prediction_dates
from serein.eval.episode_inputs import _load_predictions, _values


def _table(frame: pd.DataFrame) -> pd.DatetimeIndex:
    days = _prediction_dates(frame)
    if frame.empty:
        raise ValueError("input tables must not be empty")
    if (days.year >= 2024).any():
        raise ValueError("feature support accepts development issue dates before 2024 only")
    return days


def _features(features) -> list[str]:
    if isinstance(features, str):
        raise TypeError("features must be a sequence of column names")
    features = list(features)
    if not features or any(not isinstance(name, str) for name in features):
        raise ValueError("features must be nonempty column names")
    if len(set(features)) != len(features):
        raise ValueError("features must be nonempty and unique")
    for name in features:
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                or name in {"point_id", "issue_date", "kind"} or name.startswith("y_L")):
            raise ValueError(f"invalid input feature: {name!r}")
    return features


def _describe(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    result = {"n_rows": len(values), "n_finite": len(finite),
              "missing_rate": float(1 - len(finite) / len(values)) if len(values) else np.nan}
    for name in ("min", "q05", "q25", "median", "q75", "q95", "max", "mean", "std"):
        result[name] = np.nan
    if len(finite):
        with np.errstate(over="ignore", invalid="ignore"):
            quantiles = np.quantile(finite, [0, .05, .25, .5, .75, .95, 1])
            result.update(zip(("min", "q05", "q25", "median", "q75", "q95", "max"),
                              map(float, quantiles), strict=True))
            result.update(mean=float(finite.mean()), std=float(finite.std()))
        if not np.isfinite([*quantiles, result["mean"], result["std"]]).all():
            raise ValueError("feature statistics overflow; check input units")
    return result


def _compare(reference: np.ndarray, query: np.ndarray, ref: dict, current: dict) -> dict:
    a, b = reference, np.sort(query[np.isfinite(query)])
    result = {"status": "ok", "missing_rate_change": current["missing_rate"] - ref["missing_rate"]}
    metrics = ("ks_distance", "wasserstein", "wasserstein_ref_sd", "mean_shift_ref_sd",
               "below_reference_min_rate", "above_reference_max_rate",
               "outside_reference_90_rate")
    result.update({name: np.nan for name in metrics})
    if not len(a) or not len(b):
        result["status"] = ("no_reference_values" if not len(a) else "no_query_values")
        return result
    # At each empirical CDF jump, compare its right-continuous value in both samples.
    at_a = np.arange(1, len(a) + 1) / len(a) - np.searchsorted(b, a, side="right") / len(b)
    at_b = np.searchsorted(a, b, side="right") / len(a) - np.arange(1, len(b) + 1) / len(b)
    # Repeated values jump together, so evaluate only the last index of each tie.
    last_a = np.r_[a[1:] != a[:-1], True]
    last_b = np.r_[b[1:] != b[:-1], True]
    result["ks_distance"] = float(max(np.abs(at_a[last_a]).max(), np.abs(at_b[last_b]).max()))
    result["wasserstein"] = float(wasserstein_distance(a, b))
    result["below_reference_min_rate"] = float(np.mean(b < ref["min"]))
    result["above_reference_max_rate"] = float(np.mean(b > ref["max"]))
    result["outside_reference_90_rate"] = float(np.mean((b < ref["q05"]) | (b > ref["q95"])))
    if ref["std"] > 0:
        result["wasserstein_ref_sd"] = result["wasserstein"] / ref["std"]
        result["mean_shift_ref_sd"] = (current["mean"] - ref["mean"]) / ref["std"]
    else:
        result["status"] = "constant_reference"
    if any(np.isinf(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("feature comparisons overflow; check input units")
    return result


def analyze_support(reference: pd.DataFrame, query: pd.DataFrame, *, features,
                    groups=(), reference_scope: str = "pooled") -> dict:
    """Return univariate input shifts and row-level support, without fitting a model.

    Features must be selected before looking at query outcomes. Values are in raw
    table units. Reference scope is either pooled across all rows or matched to
    each query point. Query targets are never consulted.
    """
    reference_days, query_days = _table(reference), _table(query)
    features = _features(features)
    groups = list(groups)
    if len(set(groups)) != len(groups) or any(name not in {"point", "month"} for name in groups):
        raise ValueError("groups must be unique choices of point and month")
    if reference_scope not in {"pooled", "point"}:
        raise ValueError("reference_scope must be pooled or point")
    for name, frame in (("reference", reference), ("query", query)):
        missing = set(features) - set(frame)
        if missing:
            raise ValueError(f"{name} is missing feature columns: {', '.join(sorted(missing))}")
        if any(not pd.api.types.is_numeric_dtype(frame[column])
               or pd.api.types.is_complex_dtype(frame[column]) for column in features):
            raise ValueError(f"{name} features must have numeric dtypes")
    values = [{column: _values(frame, column) for column in features}
              for frame in (reference, query)]
    rows = query[["point_id", "issue_date"]].copy().reset_index(drop=True)
    rows["issue_date"] = query_days
    rows["seen_reference_point"] = query.point_id.isin(reference.point_id).to_numpy()
    for column in ("n_missing_features", "n_features_without_reference", "n_outside_reference_range",
                   "n_outside_reference_90"):
        rows[column] = 0
    row_groups = [("all", "all", np.arange(len(query)))]
    for group in groups:
        labels = query.point_id.to_numpy() if group == "point" else query_days.strftime("%Y-%m")
        for label in sorted(set(labels)):
            row_groups.append((group, label, np.flatnonzero(labels == label)))
    references, comparisons = [], []
    scopes = (["pooled"] if reference_scope == "pooled" else sorted(query.point_id.unique()))
    reference_indices = reference.groupby("point_id", sort=False).indices
    query_indices = query.groupby("point_id", sort=False).indices
    for scope in scopes:
        ri = (np.arange(len(reference)) if reference_scope == "pooled"
              else reference_indices.get(scope, np.array([], dtype=int)))
        qi = (np.arange(len(query)) if reference_scope == "pooled"
              else query_indices[scope])
        selected_groups = []
        for group, label, indices in row_groups:
            if reference_scope == "point":
                if group == "point" and label != scope:
                    continue
                indices = (qi if group in {"all", "point"}
                           else qi[query_days[qi].strftime("%Y-%m") == label])
            if len(indices):
                selected_groups.append((group, label, indices))
        for feature in features:
            a = values[0][feature][ri]
            ref = _describe(a)
            finite_reference_days = reference_days[ri[np.isfinite(a)]].nunique()
            a = np.sort(a[np.isfinite(a)])
            references.append({"reference_scope": reference_scope, "reference_point": scope,
                               "feature": feature, **ref,
                               "n_issue_days": reference_days[ri].nunique(),
                               "n_finite_issue_days": finite_reference_days})
            b = values[1][feature]
            finite = np.isfinite(b[qi])
            rows.loc[qi, "n_missing_features"] += ~finite
            if not ref["n_finite"]:
                rows.loc[qi, "n_features_without_reference"] += 1
            else:
                rows.loc[qi, "n_outside_reference_range"] += (
                    finite & ((b[qi] < ref["min"]) | (b[qi] > ref["max"])))
                rows.loc[qi, "n_outside_reference_90"] += (
                    finite & ((b[qi] < ref["q05"]) | (b[qi] > ref["q95"])))
            for group, label, selected in selected_groups:
                current = _describe(b[selected])
                comparisons.append({
                    "reference_scope": reference_scope, "reference_point": scope,
                    "group": group, "group_value": label, "feature": feature,
                    "n_reference_rows": len(ri), "n_reference_finite": ref["n_finite"],
                    "n_reference_days": reference_days[ri].nunique(),
                    "n_reference_finite_days": finite_reference_days,
                    "n_query_days": query_days[selected].nunique(),
                    "n_query_finite_days": query_days[selected[np.isfinite(b[selected])]].nunique(),
                    **{f"query_{name}": value for name, value in current.items()},
                    **_compare(a, b[selected], ref, current),
                })
    summaries = []
    for group, label, indices in row_groups:
        selected = rows.iloc[indices]
        summaries.append({
            "group": group, "group_value": label, "n_rows": len(selected),
            "n_issue_days": selected.issue_date.nunique(),
            "n_points": selected.point_id.nunique(),
            "n_unseen_point_rows": int((~selected.seen_reference_point).sum()),
            "n_rows_with_missing_features": int(selected.n_missing_features.gt(0).sum()),
            "n_rows_without_reference": int(selected.n_features_without_reference.gt(0).sum()),
            "n_rows_outside_reference_range": int(selected.n_outside_reference_range.gt(0).sum()),
            "n_rows_outside_reference_90": int(selected.n_outside_reference_90.gt(0).sum()),
        })
    return {"reference": pd.DataFrame(references), "shift": pd.DataFrame(comparisons),
            "rows": rows, "summary": pd.DataFrame(summaries)}


def write_support(reference_path: Path, query_path: Path, output: Path, *, features,
                  groups=(), reference_scope: str = "pooled") -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    for path in (reference_path, query_path):
        if (path.parent / "FINAL_RUN_LOCK.json").exists():
            raise ValueError("feature support accepts development inputs only")
    reference, reference_hash = _load_predictions(reference_path)
    query, query_hash = _load_predictions(query_path)
    features, groups = list(features), list(groups)
    result = analyze_support(reference, query, features=features, groups=groups,
                             reference_scope=reference_scope)
    manifest = {
        "report_version": 1, "mode": "development_input_support_only",
        "features": features, "groups": groups, "reference_scope": reference_scope,
        "sources": {name: {"path": str(path.resolve()), "sha256": digest}
                    for name, path, digest in (("reference", reference_path, reference_hash),
                                               ("query", query_path, query_hash))},
        "n_reference_rows": len(reference), "n_query_rows": len(query),
        "overlapping_point_issue_rows": int(pd.MultiIndex.from_frame(
            reference[["point_id", "issue_date"]].assign(issue_date=_table(reference))).intersection(
                pd.MultiIndex.from_frame(query[["point_id", "issue_date"]].assign(
                    issue_date=_table(query)))).size),
        "limitations": [
            "Raw univariate inputs only; joint, graph, history and transformed model support are not tested.",
            "Reference rows are equally weighted; correlated site-days are not independent samples.",
            "KS and Wasserstein distances are descriptive; no p-values or shift cutoffs are selected.",
            "Range and 5th/95th percentile flags are not confidence bounds or calibrated anomaly tests.",
            "Missing values are audited separately and are excluded from finite-value distributions.",
            "Known points and familiar inputs do not establish accuracy or interval coverage.",
            "Feature choice, reference provenance and prior use of query dates require team review.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in result.items():
        if name == "rows":
            frame.to_parquet(output / "rows.parquet", index=False)
        else:
            frame.to_csv(output / f"{name}.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature", action="append", required=True, dest="features")
    parser.add_argument("--group", action="append", choices=("point", "month"), dest="groups")
    parser.add_argument("--reference-scope", choices=("pooled", "point"), default="pooled")
    args = parser.parse_args(argv)
    try:
        write_support(args.reference, args.query, args.out, features=args.features,
                      groups=args.groups or (), reference_scope=args.reference_scope)
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as exc:
        parser.exit(2, f"feature support: {exc}\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
