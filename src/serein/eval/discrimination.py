"""Audit event ranking and fixed warning cutoffs in saved development probabilities."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from serein.aqi import USG_THRESHOLD
from serein.eval.comparison import _nonnegative_integer, _prediction_dates
from serein.eval.episode_inputs import _choices, _load_predictions, _values
from serein.eval.intervals import _groups

COUNTS = ("tp", "fp", "fn", "tn")
RATES = ("recall", "false_positive_rate", "precision", "false_alarm_ratio", "warning_rate", "csi")


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan),
                     where=denominator > 0)


def _rates(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn, tn = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    return np.stack((_divide(tp, tp + fn), _divide(fp, fp + tn), _divide(tp, tp + fp),
                     _divide(fp, tp + fp), _divide(tp + fp, tp + fp + fn + tn),
                     _divide(tp, tp + fp + fn)), axis=-1)


def _ranking(probabilities: np.ndarray, events: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Accumulate tied probabilities together; AP uses recall steps, not trapezoids."""
    n = len(events)
    positives = int(events.sum())
    negatives = n - positives
    status = "ok" if positives and negatives else (
        "no_scored_rows" if not n else "no_events" if not positives else "no_nonevents")
    summary = {"roc_auc": np.nan, "average_precision": np.nan, "ranking_status": status,
               "n_unique_probabilities": len(np.unique(probabilities))}
    order = np.argsort(-probabilities, kind="stable")
    p, y = probabilities[order], events[order]
    ends = np.flatnonzero(np.r_[p[1:] != p[:-1], True]) if n else np.array([], dtype=int)
    tp = np.r_[0, np.cumsum(y, dtype=np.int64)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    counts = np.column_stack((tp, fp, positives - tp, negatives - fp))
    curve = pd.DataFrame(counts, columns=COUNTS)
    # The first row means warn nobody, including when a forecast probability is 1.
    curve.insert(0, "cutoff", np.r_[np.nan, p[ends]])
    curve.insert(0, "warn_none", np.arange(len(curve)) == 0)
    rates = _rates(counts)
    for index, name in enumerate(RATES):
        curve[name] = rates[:, index]
    if positives and negatives:
        recall, fpr, precision = rates[:, 0], rates[:, 1], rates[:, 2]
        summary["roc_auc"] = float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))
        summary["average_precision"] = float(np.sum(np.diff(recall) * precision[1:]))
    return summary, curve


def _bounds(daily: pd.DataFrame, *, block_days: int, n_boot: int, seed: int) -> dict:
    blocks = (daily.issue_date - pd.Timestamp("1970-01-01")).dt.days // block_days
    totals = daily.groupby(blocks, sort=True)[list(COUNTS)].sum()
    totals = totals.loc[totals.sum(axis=1).gt(0)].to_numpy(float)
    result = {"n_blocks": len(totals), "n_event_blocks": int((totals[:, [0, 2]].sum(1) > 0).sum()),
              "n_nonevent_blocks": int((totals[:, [1, 3]].sum(1) > 0).sum())}
    samples = np.full((n_boot, len(RATES)), np.nan)
    status = "ok"
    if len(totals) < 5:
        status = "insufficient_blocks"
    elif not totals[:, [0, 2]].sum() or not totals[:, [1, 3]].sum():
        status = "constant_outcome"
    else:
        rng = np.random.default_rng(seed)
        for start in range(0, n_boot, 256):
            stop = min(start + 256, n_boot)
            selection = rng.integers(0, len(totals), size=(stop - start, len(totals)))
            samples[start:stop] = _rates(totals[selection].sum(axis=1))
    result["bootstrap_status"] = status
    estimates = _rates(totals.sum(axis=0))
    for index, name in enumerate(RATES):
        valid = np.isfinite(samples[:, index])
        result[f"{name}_n_boot_valid"] = int(valid.sum())
        # Conditional resampling would hide absent rare-event/forecast support.
        rate_status = status
        if status == "ok":
            rate_status = "undefined_resamples" if not valid.all() else (
                "boundary_rate" if estimates[index] in (0., 1.) else "ok")
        result[f"{name}_bootstrap_status"] = rate_status
        usable = rate_status == "ok"
        bounds = np.percentile(samples[:, index], [2.5, 97.5]) if usable else [np.nan, np.nan]
        result[f"{name}_ci_lo"], result[f"{name}_ci_hi"] = map(float, bounds)
    return result


def analyze_discrimination(
    predictions: pd.DataFrame, *, models=None, leads=None, groups=(),
    threshold: float = USG_THRESHOLD, cutoffs=(.1, .25, .5, .75, .9),
    block_days: int = 7, n_boot: int = 1000, seed: int = 0, common_rows: bool = False,
) -> dict[str, pd.DataFrame]:
    """Return exact rank curves, fixed-cutoff scores and daily confusion counts.

    Scores weight site-days equally. Every threshold is supplied in advance;
    the report never chooses an optimal cutoff or fits probabilities.
    """
    days = _prediction_dates(predictions)
    if predictions.empty or (days.year >= 2024).any():
        raise ValueError("discrimination requires nonempty development issue dates before 2024")
    if (isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("threshold must be finite and nonnegative")
    cutoffs = list(cutoffs)
    if not cutoffs or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number))
                          or np.iscomplexobj(x) or not np.isfinite(x) or not 0 <= x <= 1
                          for x in cutoffs):
        raise ValueError("cutoffs must be finite probabilities in [0, 1]")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be unique")
    cutoffs = sorted(map(float, cutoffs))
    for name, value in (("block_days", block_days), ("n_boot", n_boot), ("seed", seed)):
        _nonnegative_integer(value, name)
        if name != "seed" and not value:
            raise ValueError(f"{name} must be positive")
    if not isinstance(common_rows, (bool, np.bool_)):
        raise TypeError("common_rows must be boolean")
    groups = list(groups)
    if len(set(groups)) != len(groups) or any(g not in {"point", "month", "smoke"} for g in groups):
        raise ValueError("groups must be unique choices of point, month and smoke")
    leads = _choices(sorted(int(match[1]) for column in predictions
                           if (match := re.fullmatch(r"y_L(0|[1-9]\d*)", str(column))))
                     if leads is None else leads, "leads", integers=True)
    models = _choices(sorted({match[1] for column in predictions
                             if (match := re.fullmatch(r"(.+)_prob_L\d+", str(column)))})
                      if models is None else models, "models")
    if any(((days + pd.Timedelta(days=int(lead))).year >= 2024).any() for lead in leads):
        raise ValueError("discrimination refuses reserved target dates in 2024 and later")
    masks = list(_groups(predictions, days, groups))
    summaries, curves, thresholds, audits = [], [], [], []
    for lead in leads:
        required = [f"y_L{lead}", *[f"{model}_prob_L{lead}" for model in models]]
        missing = set(required) - set(predictions)
        if missing:
            raise ValueError(f"missing probability columns: {', '.join(sorted(missing))}")
        if any(not pd.api.types.is_numeric_dtype(predictions[column])
               or pd.api.types.is_complex_dtype(predictions[column]) for column in required):
            raise ValueError("observations and probabilities must have real numeric dtypes")
        obs = _values(predictions, f"y_L{lead}")
        observed, events = np.isfinite(obs), obs >= threshold
        probs = {model: _values(predictions, f"{model}_prob_L{lead}") for model in models}
        for model, p in probs.items():
            if ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
                raise ValueError(f"{model} probabilities must be in [0, 1] or missing")
        common = observed & np.logical_and.reduce([np.isfinite(p) for p in probs.values()])
        for model, p in probs.items():
            available = np.isfinite(p)
            pairwise = observed & available
            scored = common if common_rows else pairwise
            populations = {}
            for group, label, mask in masks:
                key = {"model": model, "lead": int(lead), "group": group, "group_value": label}
                selected = mask & scored
                n = int(selected.sum())
                n_events = int(events[selected].sum())
                ranks, curve = _ranking(p[selected], events[selected])
                population = {"n_input": int(mask.sum()), "n_observed": int((mask & observed).sum()),
                              "n_probability": int((mask & available).sum()),
                              "n_pairwise": int((mask & pairwise).sum()), "n_scored": n,
                              "n_excluded_common": int((mask & pairwise & ~scored).sum()),
                              "n_excluded": int((mask & ~scored).sum()), "n_events": n_events,
                              "n_nonevents": n - n_events,
                              "event_rate": n_events / n if n else np.nan,
                              "n_days_input": days[mask].nunique(),
                              "n_days_scored": days[selected].nunique(),
                              "n_event_days": days[selected & events].nunique(),
                              "n_nonevent_days": days[selected & ~events].nunique()}
                populations[group, label] = population
                summaries.append({**key, **population, **ranks, "threshold": float(threshold),
                                  "population": "common_rows" if common_rows else "model_rows"})
                for name, value in key.items():
                    curve[name] = value
                curves.append(curve)
            for cutoff in cutoffs:
                warning = p >= cutoff
                row = pd.DataFrame({"issue_date": days, "n_input": 1,
                                    "n_observed": observed.astype(int),
                                    "n_probability": available.astype(int),
                                    "n_pairwise": pairwise.astype(int), "n_scored": scored.astype(int),
                                    "tp": (scored & warning & events).astype(int),
                                    "fp": (scored & warning & ~events).astype(int),
                                    "fn": (scored & ~warning & events).astype(int),
                                    "tn": (scored & ~warning & ~events).astype(int)})
                for group, label, mask in masks:
                    key = {"model": model, "lead": int(lead), "group": group, "group_value": label}
                    daily = row.loc[mask].groupby("issue_date", sort=True).sum().reset_index()
                    totals = daily[list(COUNTS)].sum().to_numpy()
                    thresholds.append({**key, "cutoff": cutoff, **populations[group, label],
                                       **dict(zip(COUNTS, map(int, totals), strict=True)),
                                       **dict(zip(RATES, map(float, _rates(totals)), strict=True)),
                                       "block_days": block_days, "n_boot": n_boot, "seed": seed,
                                       **_bounds(daily, block_days=block_days, n_boot=n_boot, seed=seed)})
                    for name, value in {**key, "cutoff": cutoff}.items():
                        daily[name] = value
                    audits.append(daily)
    return {"summary": pd.DataFrame(summaries), "curves": pd.concat(curves, ignore_index=True),
            "thresholds": pd.DataFrame(thresholds), "daily": pd.concat(audits, ignore_index=True)}


def write_discrimination(source: Path, output: Path, *, plot: bool = False, **kwargs) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be boolean")
    if (source.parent / "FINAL_RUN_LOCK.json").exists():
        raise ValueError("discrimination accepts development inputs only")
    predictions, digest = _load_predictions(source)
    result = analyze_discrimination(predictions, **kwargs)
    summary, thresholds = result["summary"], result["thresholds"]
    manifest = {
        "report_version": 1, "mode": "development_discrimination_report",
        "source": str(source.resolve()), "sha256": digest, "n_input": len(predictions),
        "models": summary.model.drop_duplicates().tolist(), "leads": sorted(summary.lead.unique().tolist()),
        "threshold": float(summary.threshold.iloc[0]), "cutoffs": sorted(thresholds.cutoff.unique().tolist()),
        "groups": [x for x in summary.group.drop_duplicates() if x != "all"],
        "common_rows": bool(kwargs.get("common_rows", False)),
        "block_days": int(thresholds.block_days.iloc[0]), "block_origin": "1970-01-01",
        "n_boot": int(thresholds.n_boot.iloc[0]), "seed": int(thresholds.seed.iloc[0]),
        "plots": [{"model": model, "file": f"discrimination_{index:03d}.png"}
                  for index, model in enumerate(summary.model.drop_duplicates(), 1)] if plot else [],
        "limitations": [
            "Probabilities must describe the declared event; provenance is not inferred.",
            "No cutoff is selected and no probability is fitted or recalibrated.",
            "ROC AUC and average precision require both outcome classes here and have no confidence bounds.",
            "Average precision uses recall increments, not trapezoidal precision-recall area.",
            "Fixed-cutoff bounds resample nonempty calendar blocks, retaining every site on each issue date.",
            "Constant outcomes suppress bounds; a rate with any undefined resample also has no bounds.",
            "Rates observed at zero or one retain scores but have no percentile bounds; absent variation cannot be resampled.",
            "Bounds are exploratory and marginal; dependence between blocks and prior tuning may remain.",
            "Rows are weighted equally; groups with different event rates or availability are not directly comparable.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in result.items():
        if name in {"daily", "curves"}:
            frame.to_parquet(output / f"{name}.parquet", index=False)
        else:
            frame.to_csv(output / f"{name}.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if plot:
        from serein.eval.discrimination_plots import plot_discrimination

        for item in manifest["plots"]:
            plot_discrimination(result, item["model"], output / item["file"])
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--lead", action="append", type=int, dest="leads")
    parser.add_argument("--group", action="append", choices=("point", "month", "smoke"), dest="groups")
    parser.add_argument("--threshold", type=float, default=USG_THRESHOLD)
    parser.add_argument("--cutoff", action="append", type=float, dest="cutoffs")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--common-rows", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    settings = vars(args).copy()
    source, output = settings.pop("predictions"), settings.pop("out")
    settings["groups"] = settings["groups"] or ()
    if settings["cutoffs"] is None:
        settings.pop("cutoffs")
    try:
        write_discrimination(source, output, **settings)
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as exc:
        parser.exit(2, f"discrimination: {exc}\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
