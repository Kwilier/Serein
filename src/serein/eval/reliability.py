"""Score saved development event probabilities without selecting warning cutoffs."""

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

TOTALS = ("n_scored", "n_events", "probability_sum", "brier_sum")
METRICS = ("event_rate", "mean_probability", "brier")


def _bounds(daily: pd.DataFrame, block_days: int, n_boot: int, seed: int) -> dict:
    blocks = (daily.issue_date - pd.Timestamp("1970-01-01")).dt.days // block_days
    totals = daily.groupby(blocks, sort=True)[list(TOTALS)].sum()
    totals = totals.loc[totals.n_scored.gt(0)].to_numpy(float)
    result = {"n_blocks": len(totals), "bootstrap_status": "insufficient_blocks"}
    for name in METRICS:
        result[f"{name}_ci_lo"] = result[f"{name}_ci_hi"] = np.nan
    if len(totals) < 5:
        return result
    if totals[:, 1].sum() in (0, totals[:, 0].sum()):
        result["bootstrap_status"] = "constant_outcome"
        return result
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, len(METRICS)))
    for start in range(0, n_boot, 256):
        stop = min(start + 256, n_boot)
        selected = rng.integers(0, len(totals), size=(stop - start, len(totals)))
        sums = totals[selected].sum(axis=1)
        samples[start:stop] = sums[:, 1:] / sums[:, :1]
    bounds = np.percentile(samples, [2.5, 97.5], axis=0)
    for i, name in enumerate(METRICS):
        result[f"{name}_ci_lo"], result[f"{name}_ci_hi"] = map(float, bounds[:, i])
    result["bootstrap_status"] = "ok"
    return result


def _rates(totals) -> dict:
    n = totals["n_scored"]
    return {name: float(totals[column] / n) if n else np.nan
            for name, column in zip(METRICS, TOTALS[1:], strict=True)}


def analyze_reliability(predictions: pd.DataFrame, *, models=None, leads=None, groups=(),
                        threshold: float = USG_THRESHOLD, n_bins: int = 10,
                        block_days: int = 7, n_boot: int = 1000, seed: int = 0,
                        common_rows: bool = False) -> dict[str, pd.DataFrame]:
    """Return Brier scores, fixed-bin reliability and auditable daily totals.

    Probabilities use <model>_prob_L<lead> columns for the event y_L<lead> >=
    threshold. No probabilities or warning cutoffs are fitted. Bins are fixed
    equal-width intervals on [0, 1], with interior edges assigned to the right.
    """
    days = _prediction_dates(predictions)
    if predictions.empty or (days.year >= 2024).any():
        raise ValueError("reliability requires nonempty development issue dates before 2024")
    if (isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("threshold must be finite and nonnegative")
    for name, value in (("n_bins", n_bins), ("block_days", block_days), ("n_boot", n_boot), ("seed", seed)):
        _nonnegative_integer(value, name)
        if name != "seed" and value == 0:
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
        raise ValueError("reliability refuses reserved target dates in 2024 and later")
    masks = list(_groups(predictions, days, groups))
    edges = np.arange(n_bins + 1, dtype=float) / n_bins
    summaries, bins, audits = [], [], []
    for lead in leads:
        required = [f"y_L{lead}", *[f"{model}_prob_L{lead}" for model in models]]
        missing = set(required) - set(predictions)
        if missing:
            raise ValueError(f"missing probability columns: {', '.join(sorted(missing))}")
        if any(not pd.api.types.is_numeric_dtype(predictions[column])
               or pd.api.types.is_complex_dtype(predictions[column]) for column in required):
            raise ValueError("observations and probabilities must have real numeric dtypes")
        obs = _values(predictions, f"y_L{lead}")
        observed = np.isfinite(obs)
        events = obs >= threshold
        probs = {model: _values(predictions, f"{model}_prob_L{lead}") for model in models}
        for model, p in probs.items():
            finite = np.isfinite(p)
            if ((p[finite] < 0) | (p[finite] > 1)).any():
                raise ValueError(f"{model} probabilities must be in [0, 1] or missing")
        common = observed & np.logical_and.reduce([np.isfinite(p) for p in probs.values()])
        for model, p in probs.items():
            available = np.isfinite(p)
            eligible = observed & available
            scored = common if common_rows else eligible
            assignments = np.full(len(p), -1, dtype=int)
            assignments[available] = np.minimum(
                np.searchsorted(edges, p[available], side="right") - 1, n_bins - 1)
            row = pd.DataFrame({
                "issue_date": days, "bin": assignments, "n_input": 1,
                "n_observed": observed.astype(int), "n_probability": available.astype(int),
                "n_pairwise": eligible.astype(int), "n_scored": scored.astype(int),
                "n_events": (scored & events).astype(int),
                "probability_sum": np.where(scored, p, 0),
                "brier_sum": np.where(scored, (p - events.astype(int)) ** 2, 0),
            })
            for group, label, mask in masks:
                daily = row.loc[mask].groupby(["issue_date", "bin"], sort=True).sum().reset_index()
                totals = daily.drop(columns=["issue_date", "bin"]).sum()
                n = int(totals.n_scored)
                key = {"model": model, "lead": int(lead), "group": group, "group_value": label}
                summary = {**key, "threshold": float(threshold), "n_bins": n_bins,
                           "population": "common_rows" if common_rows else "model_rows",
                           **{name: int(totals[name]) for name in (
                               "n_input", "n_observed", "n_probability", "n_pairwise", "n_scored", "n_events")},
                           **_rates(totals), "n_days_input": daily.issue_date.nunique(),
                           "n_days_scored": daily.loc[daily.n_scored.gt(0), "issue_date"].nunique(),
                           "n_boot": n_boot, "seed": seed, "block_days": block_days,
                           "status": "ok" if n else "no_scored_rows",
                           **_bounds(daily, block_days, n_boot, seed)}
                summary["n_excluded_common"] = summary["n_pairwise"] - n
                summary["n_excluded"] = summary["n_input"] - n
                summary["probability_bias"] = summary["mean_probability"] - summary["event_rate"]
                rel = resolution = ece = 0.
                for index in range(-1, n_bins):
                    subset = daily.loc[daily.bin.eq(index)]
                    total = subset.drop(columns=["issue_date", "bin"]).sum()
                    rates = _rates(total)
                    count = int(total.n_scored)
                    if count:
                        weight = count / n
                        gap = rates["mean_probability"] - rates["event_rate"]
                        rel += weight * gap ** 2
                        resolution += weight * (rates["event_rate"] - summary["event_rate"]) ** 2
                        ece += weight * abs(gap)
                    bins.append({**key, "bin": index,
                                 "bin_lo": edges[index] if index >= 0 else np.nan,
                                 "bin_hi": edges[index + 1] if index >= 0 else np.nan,
                                 **{name: int(total[name]) for name in (
                                     "n_input", "n_observed", "n_probability", "n_pairwise", "n_scored", "n_events")},
                                 **rates, "probability_bias": rates["mean_probability"] - rates["event_rate"],
                                 "n_days_scored": subset.loc[subset.n_scored.gt(0), "issue_date"].nunique(),
                                 **_bounds(subset, block_days, n_boot, seed)})
                uncertainty = summary["event_rate"] * (1 - summary["event_rate"])
                binned = rel - resolution + uncertainty
                summary.update(reliability=rel if n else np.nan,
                               resolution=resolution if n else np.nan, uncertainty=uncertainty,
                               brier_binned=binned, brier_grouping_difference=summary["brier"] - binned,
                               ece=ece if n else np.nan)
                summaries.append(summary)
                for name, value in key.items():
                    daily[name] = value
                audits.append(daily)
    return {"summary": pd.DataFrame(summaries), "bins": pd.DataFrame(bins),
            "daily": pd.concat(audits, ignore_index=True)}


def write_reliability(source: Path, output: Path, *, plot: bool = False, **kwargs) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not isinstance(plot, (bool, np.bool_)):
        raise TypeError("plot must be boolean")
    if (source.parent / "FINAL_RUN_LOCK.json").exists():
        raise ValueError("reliability accepts development inputs only")
    predictions, digest = _load_predictions(source)
    result = analyze_reliability(predictions, **kwargs)
    summary = result["summary"]
    manifest = {
        "report_version": 1, "source": str(source.resolve()), "sha256": digest,
        "mode": "development_probability_report", "threshold": float(summary.threshold.iloc[0]),
        "models": summary.model.drop_duplicates().tolist(), "leads": sorted(summary.lead.unique().tolist()),
        "n_bins": int(summary.n_bins.iloc[0]), "n_boot": int(summary.n_boot.iloc[0]),
        "seed": int(summary.seed.iloc[0]), "block_days": int(summary.block_days.iloc[0]),
        "groups": [group for group in summary.group.drop_duplicates() if group != "all"],
        "common_rows": bool(kwargs.get("common_rows", False)),
        "block_origin": "1970-01-01", "n_input": len(predictions),
        "plots": [{"model": model, "file": f"reliability_{index:03d}.png"}
                  for index, model in enumerate(summary.model.drop_duplicates(), 1)] if plot else [],
        "limitations": [
            "Probabilities must already describe the declared event threshold; this is not inferred.",
            "CQR interval coverage does not establish probability calibration.",
            "Bootstrap bounds are exploratory, marginal and not adjusted for multiple groups or bins.",
            "Constant event outcomes suppress bootstrap bounds; unobserved event variation cannot be resampled.",
            "Only fixed calendar blocks with scored rows are resampled; dependence between blocks may remain.",
            "Binned decomposition is exact for bin-mean forecasts, not for the raw Brier score.",
            "Reference periods, prior tuning and probability provenance require team review.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in result.items():
        if name == "daily":
            frame.to_parquet(output / "daily.parquet", index=False)
        else:
            frame.to_csv(output / f"{name}.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if plot:
        from serein.eval.reliability_plots import plot_reliability

        for item in manifest["plots"]:
            plot_reliability(result["bins"], item["model"], output / item["file"])
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--lead", action="append", type=int, dest="leads")
    parser.add_argument("--group", action="append", choices=("point", "month", "smoke"), dest="groups")
    parser.add_argument("--threshold", type=float, default=USG_THRESHOLD)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--common-rows", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        write_reliability(args.predictions, args.out, plot=args.plot, models=args.models, leads=args.leads,
                          groups=args.groups or (), threshold=args.threshold, n_bins=args.n_bins,
                          block_days=args.block_days, n_boot=args.n_boot, seed=args.seed,
                          common_rows=args.common_rows)
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as exc:
        parser.exit(2, f"reliability: {exc}\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
