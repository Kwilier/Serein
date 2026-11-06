"""Paired, day-level forecast comparisons from saved backtest predictions.

    uv run python -m serein.eval.comparison outputs/backtest/west_us_dev
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from serein.eval.metrics import bootstrap_rmse_diff

BASELINES = ("noaa_best", "aqm_bc", "hrrr_smoke", "persistence", "climatology",
             "gbm", "smokecast_nograph")


def _calendar_dates(values) -> pd.DatetimeIndex:
    raw = np.asarray(values)
    if raw.ndim != 1 or (raw.size and pd.api.types.is_numeric_dtype(raw.dtype)):
        raise ValueError("issue dates must be a one-dimensional sequence of calendar dates")
    try:
        days = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("issue dates must be valid calendar dates") from exc
    if days.hasnans or days.tz is not None or not days.equals(days.normalize()):
        raise ValueError("issue dates must be nonmissing, timezone-free dates at midnight")
    return days


def _nonnegative_integer(value, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def diebold_mariano(daily_difference, dates, max_lag: int = 0) -> dict:
    """Two-sided asymptotic DM test, with Bartlett calendar-day covariances.

    Differences are daily mean loss(model) - loss(baseline). Negative favors
    the model. Missing calendar days contribute no covariance pairs; they are
    never compressed into adjacent observations. No small-sample correction
    is applied. See docs/FORECAST_COMPARISON.md for assumptions and formulas.
    """
    max_lag = _nonnegative_integer(max_lag, "max_lag")
    difference = np.asarray(daily_difference, dtype=float)
    days = _calendar_dates(dates)
    if difference.ndim != 1 or difference.shape != days.shape:
        raise ValueError("daily differences and dates must have the same one-dimensional shape")
    if not np.isfinite(difference).all():
        raise ValueError("daily differences must be finite")
    if days.has_duplicates:
        raise ValueError("daily differences require unique issue dates")
    order = np.argsort(days)
    days, difference = days[order], difference[order]
    n = len(difference)
    scale = np.max(np.abs(difference)) if n else 0.
    mean = float((difference / scale).mean() * scale) if scale else 0.
    result = {"n_days": n, "max_lag": max_lag,
              "mean_loss_diff": mean if n else np.nan,
              "dm_stat": np.nan, "p_two_sided": np.nan,
              "status": "insufficient_days"}
    if n < max(5, max_lag + 2):
        return result

    # Scaling leaves the statistic unchanged and avoids squaring large losses.
    if scale == 0:
        result["status"] = "zero_variance"
        return result
    scaled = difference / scale
    centered = scaled - scaled.mean()
    variance_sum = float(centered @ centered)
    for lag in range(1, max_lag + 1):
        prior = days.get_indexer(days - pd.Timedelta(days=lag))
        paired = prior >= 0
        covariance_sum = centered[paired] @ centered[prior[paired]]
        variance_sum += 2 * (1 - lag / (max_lag + 1)) * covariance_sum
    if not np.isfinite(variance_sum) or variance_sum <= 0:
        result["status"] = "zero_variance"
        return result
    statistic = float(scaled.mean() * n / np.sqrt(variance_sum))
    result.update(dm_stat=statistic, p_two_sided=float(2 * norm.sf(abs(statistic))),
                  status="ok")
    return result


def holm_adjust(p_values) -> np.ndarray:
    """Holm adjustment over the supplied family, retaining unavailable tests."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.isinf(p).any() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a vector of probabilities or NaN")
    adjusted = np.full(p.shape, np.nan)
    order = np.flatnonzero(np.isfinite(p))
    order = order[np.argsort(p[order], kind="stable")]
    # An unavailable planned comparison still counts toward the family size.
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * (len(p) - np.arange(len(order)))
    ))
    return adjusted


def compare_predictions(
    predictions: pd.DataFrame, *, model: str = "smokecast",
    baselines=None, leads=None, max_lag: int | None = None,
    n_boot: int = 1000, seed: int = 0, subsets=None, common_rows: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare selected subsets with one multiplicity family across the report.

    Days receive equal weight in the test. The accompanying RMSE values use
    paired site-days, matching the existing backtest's row-weighted estimand.
    """
    _prediction_dates(predictions)
    subsets = ["all"] if subsets is None else list(subsets)
    if (not subsets or any(name not in ("all", "smoke_yesterday") for name in subsets)
            or len(set(subsets)) != len(subsets)):
        raise ValueError("subsets must be unique choices of all or smoke_yesterday")
    if not isinstance(common_rows, (bool, np.bool_)):
        raise TypeError("common_rows must be a boolean")
    baselines = None if baselines is None else list(baselines)
    leads = None if leads is None else list(leads)
    smoke, n_unknown = None, 0
    if "smoke_yesterday" in subsets:
        if "hms_prev" not in predictions:
            raise ValueError("smoke_yesterday requires the hms_prev column")
        hms = pd.to_numeric(predictions.hms_prev, errors="raise").to_numpy(
            dtype=float, na_value=np.nan,
        )
        known = np.isfinite(hms)
        if np.isinf(hms).any() or not np.isin(hms[known], [0, 1, 2, 3]).all():
            raise ValueError("hms_prev must be 0, 1, 2, 3 or missing")
        smoke = known & (hms >= 1)
        n_unknown = int((~known).sum())
    summaries, audits = [], []
    for subset in subsets:
        selected = predictions if subset == "all" else predictions.loc[smoke]
        summary, daily = _compare_predictions(
            selected, model=model, baselines=baselines, leads=leads, max_lag=max_lag,
            n_boot=n_boot, seed=seed, common_rows=common_rows,
        )
        summary.insert(0, "subset", subset)
        daily.insert(0, "subset", subset)
        summary["n_total"] = len(predictions)
        summary["n_subset_excluded"] = len(predictions) - len(selected)
        summary["n_subset_unknown"] = n_unknown if subset == "smoke_yesterday" else 0
        summaries.append(summary)
        audits.append(daily)
    summary = pd.concat(summaries, ignore_index=True)
    summary["p_two_sided_holm"] = holm_adjust(summary.p_two_sided)
    summary["family_size"] = len(summary)
    return summary, pd.concat(audits, ignore_index=True)


def _prediction_dates(predictions: pd.DataFrame) -> pd.DatetimeIndex:
    if not predictions.columns.is_unique:
        raise ValueError("prediction columns must be unique")
    required = {"issue_date", "point_id"}
    if not required.issubset(predictions.columns):
        raise ValueError("predictions require issue_date and point_id columns")
    days = _calendar_dates(predictions["issue_date"])
    ids = predictions["point_id"]
    if ids.isna().any() or not ids.map(lambda value: isinstance(value, str) and bool(
            value.strip())).all():
        raise ValueError("point_id must contain nonempty strings")
    keys = pd.DataFrame({"issue_date": days, "point_id": ids.to_numpy()})
    if keys.duplicated().any():
        raise ValueError("predictions require unique (issue_date, point_id) rows")
    return days


def _compare_predictions(
    predictions: pd.DataFrame, *, model: str, baselines, leads, max_lag,
    n_boot: int, seed: int, common_rows: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = _prediction_dates(predictions)
    if max_lag is not None:
        max_lag = _nonnegative_integer(max_lag, "max_lag")
    n_boot = _nonnegative_integer(n_boot, "n_boot")
    seed = _nonnegative_integer(seed, "seed")
    if n_boot == 0:
        raise ValueError("n_boot must be a positive integer")
    if leads is None:
        leads = sorted(int(match[1]) for column in predictions.columns
                       if (match := re.fullmatch(r"y_L(\d+)", str(column))))
    else:
        leads = list(leads)
    if not leads:
        raise ValueError("at least one lead is required")
    leads = [_nonnegative_integer(lead, "lead") for lead in leads]
    if len(set(leads)) != len(leads):
        raise ValueError("leads must be unique")
    if baselines is None:
        baselines = [name for name in BASELINES if any(
            f"{name}_L{lead}" in predictions for lead in leads)]
    else:
        baselines = list(baselines)
    if (not baselines or len(set(baselines)) != len(baselines)
            or model in baselines or not isinstance(model, str) or not model.strip()
            or any(not isinstance(name, str) or not name.strip() for name in baselines)):
        raise ValueError("choose a model and unique, distinct baseline names")

    summaries, daily_tables = [], []
    for lead in leads:
        shared = np.ones(len(predictions), dtype=bool)
        if common_rows:
            columns = [f"{name}_L{lead}" for name in [model, *baselines, "y"]]
            missing = [column for column in columns if column not in predictions]
            if missing:
                raise ValueError(f"missing prediction columns: {', '.join(missing)}")
            shared = np.isfinite(predictions[columns].to_numpy(dtype=float, na_value=np.nan)).all(
                axis=1,
            )
        for baseline in baselines:
            columns = [f"{model}_L{lead}", f"{baseline}_L{lead}", f"y_L{lead}"]
            missing = [column for column in columns if column not in predictions]
            if missing:
                raise ValueError(f"missing prediction columns: {', '.join(missing)}")
            values = predictions[columns].to_numpy(dtype=float, na_value=np.nan)
            pairwise = np.isfinite(values).all(axis=1)
            paired = pairwise & shared
            a, b, observed = values[paired].T
            with np.errstate(over="ignore", invalid="ignore"):
                loss_a, loss_b = (a - observed) ** 2, (b - observed) ** 2
            if not np.isfinite(loss_a).all() or not np.isfinite(loss_b).all():
                raise ValueError("squared forecast errors must be finite")
            daily = pd.DataFrame({"issue_date": days,
                                  "loss_model": np.nan, "loss_baseline": np.nan})
            loss_scale = max(loss_a.max(initial=0.), loss_b.max(initial=0.)) or 1.
            daily.loc[paired, "loss_model"] = loss_a / loss_scale
            daily.loc[paired, "loss_baseline"] = loss_b / loss_scale
            grouped = daily.groupby("issue_date", sort=True)
            means = grouped[["loss_model", "loss_baseline"]].mean() * loss_scale
            means["n_paired"] = grouped.loss_model.count()
            means["n_input"] = grouped.size()
            means["n_excluded"] = means.n_input - means.n_paired
            means["loss_diff"] = means.loss_model - means.loss_baseline
            lag = max(lead - 1, 0) if max_lag is None else max_lag
            scoreable = means.n_paired > 0
            test = diebold_mariano(means.loc[scoreable, "loss_diff"],
                                  means.index[scoreable], max_lag=lag)
            # Bootstrap whole days but retain site-day weighting within each draw.
            rmse_scale = np.sqrt(loss_scale)
            bootstrap = bootstrap_rmse_diff(
                (a - observed) / rmse_scale, (b - observed) / rmse_scale,
                np.zeros_like(observed), days[paired].to_numpy(), n_boot=n_boot, seed=seed,
            )
            rmse_a, rmse_b = _rmse_from_losses(loss_a), _rmse_from_losses(loss_b)
            summaries.append({
                "model": model, "baseline": baseline, "lead": lead,
                "method": "dm_bartlett_normal", "loss": "squared_error_equal_day",
                "population": "common_rows" if common_rows else "pairwise",
                "n_input": len(predictions), "n_paired": int(paired.sum()),
                "n_excluded": int((~paired).sum()),
                "n_pairwise_available": int(pairwise.sum()),
                "n_excluded_common": int((pairwise & ~shared).sum()),
                "n_days_input": len(means), "n_days_excluded": int((~scoreable).sum()),
                "rmse_model": rmse_a, "rmse_baseline": rmse_b,
                "rmse_diff": rmse_a - rmse_b,
                "rmse_diff_ci_lo": bootstrap.get("ci_lo", np.nan) * rmse_scale,
                "rmse_diff_ci_hi": bootstrap.get("ci_hi", np.nan) * rmse_scale,
                "bootstrap_status": "ok" if bootstrap else "insufficient_days",
                "n_boot": n_boot, "bootstrap_seed": seed,
                **test,
            })
            means = means.reset_index()
            means["model"], means["baseline"], means["lead"] = model, baseline, lead
            daily_tables.append(means)
    summary = pd.DataFrame(summaries)
    return summary, pd.concat(daily_tables, ignore_index=True)


def _rmse_from_losses(losses: np.ndarray) -> float:
    if not len(losses):
        return np.nan
    scale = losses.max()
    return float(np.sqrt(scale) * np.sqrt(np.mean(losses / scale))) if scale else 0.


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="backtest directory or predictions parquet file")
    parser.add_argument("--model", default="smokecast")
    parser.add_argument("--baseline", action="append", dest="baselines")
    parser.add_argument("--lead", action="append", type=int, dest="leads")
    parser.add_argument("--max-lag", type=int, help="calendar-day HAC lag; default: max(lead - 1, 0)")
    parser.add_argument("--n-boot", type=int, default=1000, help="whole-day bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap random seed")
    parser.add_argument("--subset", action="append", dest="subsets",
                        choices=("all", "smoke_yesterday"), help="repeat to include both subsets")
    parser.add_argument("--common-rows", action="store_true",
                        help="use rows finite for every selected baseline at each lead")
    parser.add_argument("--output", type=Path, help="new summary CSV; default: stdout")
    parser.add_argument("--daily-output", type=Path, help="new daily loss CSV for auditing")
    args = parser.parse_args(argv)
    source = args.run / "predictions.parquet" if args.run.is_dir() else args.run
    try:
        summary, daily = compare_predictions(
            pd.read_parquet(source), model=args.model, baselines=args.baselines,
            leads=args.leads, max_lag=args.max_lag, n_boot=args.n_boot, seed=args.seed,
            subsets=args.subsets, common_rows=args.common_rows,
        )
        destinations = [path.resolve() for path in (args.output, args.daily_output)
                        if path is not None]
        if len(set(destinations)) != len(destinations):
            raise ValueError("summary and daily output paths must differ")
        for path in destinations:
            if path.exists():
                raise ValueError(f"output already exists: {path}")
        for path, table in ((args.output, summary), (args.daily_output, daily)):
            if path is not None:
                with path.open("x", encoding="utf-8", newline="") as stream:
                    table.to_csv(stream, index=False)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is None:
        print(summary.to_csv(index=False, lineterminator="\n"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
