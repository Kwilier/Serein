"""Audit interval coverage in saved development forecasts without refitting."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from serein.eval.comparison import _nonnegative_integer, _prediction_dates
from serein.eval.episode_inputs import _choices, _load_predictions, _values

METRICS = ("coverage", "lower_miss_rate", "upper_miss_rate", "mean_width", "interval_score")
TOTALS = ("n_scored", "n_covered", "n_lower_miss", "n_upper_miss", "width_sum", "score_sum")


def _bootstrap(daily: pd.DataFrame, *, block_days: int, n_boot: int, seed: int) -> dict:
    # Fixed calendar blocks preserve gaps and cluster every site on an issue day.
    block = ((daily.issue_date - pd.Timestamp("1970-01-01")).dt.days // block_days)
    totals = daily.groupby(block, sort=True)[list(TOTALS)].sum()
    totals = totals.loc[totals.n_scored.gt(0)].to_numpy(dtype=float)
    result = {"n_blocks": len(totals), "bootstrap_status": "insufficient_blocks"}
    for name in METRICS:
        result[f"{name}_ci_lo"] = result[f"{name}_ci_hi"] = np.nan
    if len(totals) < 5:
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


def _groups(predictions: pd.DataFrame, days: pd.DatetimeIndex, groups: list[str]):
    yield "all", "all", np.ones(len(predictions), dtype=bool)
    for name in groups:
        if name == "point":
            values = predictions.point_id.to_numpy()
        elif name == "month":
            values = days.strftime("%Y-%m").to_numpy()
        else:
            if "hms_prev" not in predictions:
                raise ValueError("smoke grouping requires hms_prev")
            hms = _values(predictions, "hms_prev")
            known = np.isfinite(hms)
            if not np.isin(hms[known], [0, 1, 2, 3]).all():
                raise ValueError("hms_prev must be 0, 1, 2, 3 or missing")
            values = np.where(~known, "unknown", np.where(hms >= 1, "smoke", "clear"))
        for value in sorted(set(values)):
            yield name, value, values == value


def analyze_intervals(
    predictions: pd.DataFrame, *, models=None, leads=None, groups=(),
    alpha: float = .1, block_days: int = 7, n_boot: int = 1000, seed: int = 0,
    common_rows: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-weighted scores and daily sufficient statistics for every group.

    Each model requires lower and upper columns, not a point prediction. Missing
    bounds remain in the audit. Reversed finite bounds fail even without a target.
    Groups describe existing rows only, not unexported sites or days.
    """
    days = _prediction_dates(predictions)
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be finite and in (0, 1)")
    for name, value in (("block_days", block_days), ("n_boot", n_boot), ("seed", seed)):
        _nonnegative_integer(value, name)
        if name != "seed" and value == 0:
            raise ValueError(f"{name} must be positive")
    if not isinstance(common_rows, (bool, np.bool_)):
        raise TypeError("common_rows must be boolean")
    groups = list(groups)
    if len(set(groups)) != len(groups) or any(g not in {"point", "month", "smoke"} for g in groups):
        raise ValueError("groups must be unique choices of point, month, smoke")
    leads = _choices(sorted(int(match[1]) for column in predictions
                           if (match := re.fullmatch(r"y_L(0|[1-9]\d*)", str(column))))
                     if leads is None else leads, "leads", integers=True)
    models = _choices(sorted({match[1] for column in predictions
                             if (match := re.fullmatch(r"(.+)_lo_L\d+", str(column)))})
                      if models is None else models, "models")
    masks = list(_groups(predictions, days, groups))
    summaries, audits = [], []
    for lead in leads:
        required = [f"y_L{lead}"] + [f"{model}_{side}_L{lead}"
                                     for model in models for side in ("lo", "hi")]
        missing = sorted(set(required) - set(predictions))
        if missing:
            raise ValueError(f"missing interval columns: {', '.join(missing)}")
        values = {column: _values(predictions, column) for column in required}
        obs = values[f"y_L{lead}"]
        observed = np.isfinite(obs)
        bounds, available = {}, {}
        for model in models:
            lo, hi = (values[f"{model}_{side}_L{lead}"] for side in ("lo", "hi"))
            paired = np.isfinite(lo) & np.isfinite(hi)
            if np.any(lo[paired] > hi[paired]):
                raise ValueError(f"{model} lower bounds exceed upper bounds at lead {lead}")
            bounds[model], available[model] = (lo, hi), paired
        common = observed & np.logical_and.reduce(list(available.values()))
        for model in models:
            lo, hi = bounds[model]
            eligible = observed & available[model]
            scored = common if common_rows else eligible
            lower, upper = scored & (obs < lo), scored & (obs > hi)
            covered = scored & ~lower & ~upper
            width, score = np.zeros(len(obs)), np.zeros(len(obs))
            with np.errstate(over="ignore", invalid="ignore"):
                width[scored] = hi[scored] - lo[scored]
                score[scored] = width[scored]
                score[lower] += (2 / alpha) * (lo[lower] - obs[lower])
                score[upper] += (2 / alpha) * (obs[upper] - hi[upper])
            if not np.isfinite(width).all() or not np.isfinite(score).all():
                raise ValueError("interval widths or scores overflow; check concentration units")
            row = pd.DataFrame({
                "issue_date": days, "n_input": 1, "n_observed": observed.astype(int),
                "n_bounds": available[model].astype(int), "n_pairwise": eligible.astype(int),
                "n_scored": scored.astype(int), "n_covered": covered.astype(int),
                "n_lower_miss": lower.astype(int), "n_upper_miss": upper.astype(int),
                "width_sum": width, "score_sum": score,
            })
            for group, label, mask in masks:
                daily = row.loc[mask].groupby("issue_date", sort=True).sum().reset_index()
                n = int(daily.n_scored.sum())
                totals = daily[list(TOTALS)].sum().to_numpy(dtype=float)
                metrics = totals[1:] / n if n else np.full(len(METRICS), np.nan)
                if not np.isfinite(totals).all():
                    raise ValueError("interval score totals overflow; check concentration units")
                key = {"model": model, "lead": int(lead), "group": group, "group_value": label}
                summary = {**key, "alpha": alpha, "nominal_coverage": 1 - alpha,
                           "population": "common_rows" if common_rows else "model_rows",
                           "n_input": int(daily.n_input.sum()),
                           "n_observed": int(daily.n_observed.sum()),
                           "n_bounds": int(daily.n_bounds.sum()),
                           "n_pairwise": int(daily.n_pairwise.sum()), "n_scored": n,
                           "n_days_input": len(daily),
                           "n_days_scored": int(daily.n_scored.gt(0).sum()),
                           "n_boot": n_boot, "seed": seed, "block_days": block_days,
                           "status": "ok" if n else "no_scored_rows",
                           **dict(zip(METRICS, map(float, metrics), strict=True)),
                           **_bootstrap(daily, block_days=block_days, n_boot=n_boot, seed=seed)}
                summary["n_excluded"] = summary["n_input"] - n
                summary["n_excluded_common"] = summary["n_pairwise"] - n
                summaries.append(summary)
                for name, value in key.items():
                    daily[name] = value
                audits.append(daily)
    return pd.DataFrame(summaries), pd.concat(audits, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="development run directory or predictions parquet")
    parser.add_argument("--config", type=Path, default=Path("configs/west_us.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--lead", action="append", type=int, dest="leads")
    parser.add_argument("--group", action="append", dest="groups", choices=("point", "month", "smoke"))
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--common-rows", action="store_true")
    parser.add_argument("--plot", action="store_true", help="write one descriptive coverage PNG per model")
    args = parser.parse_args(argv)
    source = args.run / "predictions.parquet" if args.run.is_dir() else args.run
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        if not args.output.parent.is_dir():
            raise ValueError("output parent must exist")
        if (source.parent / "FINAL_RUN_LOCK.json").exists():
            raise ValueError("interval reports accept development runs only")
        predictions, digest = _load_predictions(source)
        config_bytes = args.config.read_bytes()
        config = yaml.safe_load(io.BytesIO(config_bytes))
        years = config["split"]["calibrate"]
        days = _prediction_dates(predictions)
        if not days.year.isin(years).all():
            raise ValueError("issue dates must belong to configured calibration years")
        summary, daily = analyze_intervals(
            predictions, models=args.models, leads=args.leads, groups=args.groups or (),
            alpha=args.alpha, block_days=args.block_days, n_boot=args.n_boot, seed=args.seed,
            common_rows=args.common_rows,
        )
        for lead in summary.lead.unique():
            if not (days + pd.Timedelta(days=int(lead))).year.isin(years).all():
                raise ValueError("target dates must belong to configured calibration years")
        manifest = {"report_version": 1, "source": str(source.resolve()), "sha256": digest,
                    "config": str(args.config.resolve()),
                    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "development_years": years, "n_input": len(predictions),
                    "models": summary.model.drop_duplicates().tolist(),
                    "leads": summary.lead.drop_duplicates().tolist(),
                    "groups": args.groups or [], "alpha": args.alpha,
                    "block_days": args.block_days, "block_origin": "1970-01-01",
                    "n_boot": args.n_boot, "seed": args.seed, "common_rows": args.common_rows,
                    "plots": [{"model": model, "file": f"coverage_{index:03d}.png"}
                              for index, model in enumerate(summary.model.drop_duplicates(), start=1)]
                    if args.plot else [],
                    "interpretation": "exploratory development coverage; no conditional or spatial guarantee"}
        args.output.mkdir(exist_ok=False)
        summary.to_csv(args.output / "summary.csv", index=False)
        daily.to_parquet(args.output / "daily.parquet", index=False)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if args.plot:
            from serein.eval.interval_plots import plot_coverage

            for item in manifest["plots"]:
                plot_coverage(summary, item["model"], args.output / item["file"])
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.output}: {len(summary)} interval summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
