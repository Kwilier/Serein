"""Audit high-PM episodes in saved development forecasts without fitting models."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from serein.aqi import USG_THRESHOLD
from serein.config import load_study
from serein.eval.comparison import BASELINES, _prediction_dates
from serein.eval.episode_inputs import _choices, _load_predictions, _observations, _values

KEYS = ["point_id", "valid_date"]
CATALOG_COLUMNS = ["episode_id", "point_id", "start_date", "end_date", "n_days",
                   "observed_peak", "observed_peak_date", "left_censored", "right_censored"]
SCORE_COLUMNS = ["episode_id", "model", "lead", "n_forecasts", "status", "predicted_peak",
                 "predicted_peak_date", "peak_ratio", "peak_bias", "peak_timing_error_days",
                 "prediction_at_observed_peak", "observed_peak_ratio", "first_warning_date",
                 "first_warning_issue_date", "onset_delay_days", "warning_notice_days",
                 "onset_warned", "detected", "n_intervals", "interval_coverage",
                 "observed_peak_covered"]


@dataclass
class EpisodeReport:
    catalog: pd.DataFrame
    scores: pd.DataFrame
    summary: pd.DataFrame
    daily: pd.DataFrame


def _catalog(observations: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    episodes, membership = [], []
    for point_id, group in observations.groupby("point_id", sort=True, observed=True):
        known = group.loc[group.observed.notna()].sort_values("valid_date")
        high = known.loc[known.observed >= threshold].copy()
        if high.empty:
            continue
        breaks = high.valid_date.diff().ne(pd.Timedelta(days=1)).cumsum()
        starts = high.groupby(breaks).valid_date.transform("first")
        high["episode_id"] = str(point_id) + ":" + starts.dt.strftime("%Y%m%d")
        membership.append(high[[*KEYS, "episode_id"]])
        observed_days = known.set_index("valid_date").observed
        for _, event in high.groupby(breaks, sort=True):
            start, end = event.valid_date.iloc[0], event.valid_date.iloc[-1]
            peak = event.loc[event.observed.idxmax()]
            previous = observed_days.get(start - pd.Timedelta(days=1), np.nan)
            following = observed_days.get(end + pd.Timedelta(days=1), np.nan)
            episodes.append({
                "episode_id": f"{point_id}:{start:%Y%m%d}", "point_id": point_id,
                "start_date": start, "end_date": end, "n_days": len(event),
                "observed_peak": float(peak.observed), "observed_peak_date": peak.valid_date,
                "left_censored": not np.isfinite(previous),
                "right_censored": not np.isfinite(following),
            })
    members = (pd.concat(membership, ignore_index=True) if membership else
               observations.iloc[:0][KEYS].assign(episode_id=pd.Series(dtype="object")))
    return pd.DataFrame(episodes, columns=CATALOG_COLUMNS), members


def _score_episode(event, rows: pd.DataFrame, model: str, lead: int,
                   threshold: float) -> dict:
    finite = rows.predicted.notna()
    available = rows.loc[finite]
    n = len(available)
    result = dict.fromkeys(SCORE_COLUMNS, np.nan)
    result.update(episode_id=event.episode_id, model=model, lead=lead, n_forecasts=n,
                  status="complete" if n == event.n_days else "partial" if n else "missing",
                  predicted_peak_date=pd.NaT, first_warning_date=pd.NaT, first_warning_issue_date=pd.NaT,
                  n_intervals=int(rows.interval_available.sum()))
    if n:
        peak = available.loc[available.predicted.idxmax()]
        warnings = available.loc[available.predicted >= threshold]
        result.update(predicted_peak=float(peak.predicted), predicted_peak_date=peak.valid_date,
                      peak_ratio=float(peak.predicted / event.observed_peak),
                      peak_bias=float(peak.predicted - event.observed_peak),
                      peak_timing_error_days=int((peak.valid_date - event.observed_peak_date).days),
                      detected=bool(len(warnings)))
        if len(warnings):
            first = warnings.valid_date.min()
            issued = first - pd.Timedelta(days=int(lead))
            result.update(first_warning_date=first, first_warning_issue_date=issued,
                          onset_delay_days=int((first - event.start_date).days),
                          warning_notice_days=int((event.start_date - issued).days))
    at_onset = rows.loc[rows.valid_date.eq(event.start_date)].iloc[0]
    if pd.notna(at_onset.predicted):
        result["onset_warned"] = bool(at_onset.predicted >= threshold)
    at_peak = rows.loc[rows.valid_date.eq(event.observed_peak_date)].iloc[0]
    if pd.notna(at_peak.predicted):
        result.update(prediction_at_observed_peak=float(at_peak.predicted),
                      observed_peak_ratio=float(at_peak.predicted / event.observed_peak))
    intervals = rows.loc[rows.interval_available]
    if len(intervals):
        result["interval_coverage"] = float(intervals.covered.mean())
    if at_peak.interval_available:
        result["observed_peak_covered"] = bool(at_peak.covered)
    return result


def analyze_episodes(predictions: pd.DataFrame, *, models=None, leads=None,
                     threshold: float = USG_THRESHOLD, common_episodes: bool = True,
                     uncensored_only: bool = False) -> EpisodeReport:
    """Detect contiguous observed high-PM days, then audit each model and lead.

    Missing dates break episodes. Summaries score complete episodes only, using
    the same episode population across selected models at each lead by default.
    Partial forecasts remain in the detailed audit, never as complete misses.
    """
    days = _prediction_dates(predictions)
    if isinstance(threshold, (bool, np.bool_)) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not isinstance(common_episodes, bool) or not isinstance(uncensored_only, bool):
        raise TypeError("population options must be booleans")
    observations = _observations(predictions, days)
    indexed_observations = observations.set_index(KEYS).observed
    # Rounding tolerance must never reconcile two different episode classifications.
    for column in predictions:
        if match := re.fullmatch(r"y_L(\d+)", str(column)):
            original = _values(predictions, column)
            keys = pd.MultiIndex.from_arrays([
                predictions.point_id, days + pd.Timedelta(days=int(match[1]))])
            canonical = indexed_observations.reindex(keys).to_numpy()
            known = np.isfinite(original)
            if ((original[known] >= threshold) != (canonical[known] >= threshold)).any():
                raise ValueError("conflicting observations across the episode threshold")
    available_leads = sorted(int(match[1]) for column in predictions
                             if (match := re.fullmatch(r"y_L(\d+)", str(column))))
    leads = _choices(available_leads if leads is None else leads, "leads", integers=True)
    if models is None:
        models = [model for model in ("smokecast", *BASELINES) if all(
            f"{model}_L{lead}" in predictions for lead in leads)]
    models = _choices(models, "models")
    for lead in leads:
        if lead not in available_leads:
            raise ValueError(f"missing observations for lead {lead}")
    catalog, membership = _catalog(observations, float(threshold))
    daily_parts, scores = [], []
    # This join retains absent issue rows inside an observed episode as missing forecasts.
    observed = observations.loc[observations.observed.notna()].copy()
    if observed.empty:
        raise ValueError("at least one finite observation is required")
    observed = observed.merge(membership, on=KEYS, how="left", validate="one_to_one")
    for lead in leads:
        for model in models:
            column = f"{model}_L{lead}"
            if column not in predictions:
                raise ValueError(f"missing prediction column: {column}")
            forecast = pd.DataFrame({"point_id": predictions.point_id.to_numpy(),
                                     "valid_date": days + pd.Timedelta(days=int(lead)),
                                     "predicted": _values(predictions, column)})
            lo, hi = f"{model}_lo_L{lead}", f"{model}_hi_L{lead}"
            if (lo in predictions) != (hi in predictions):
                raise ValueError(f"interval columns must include both {lo} and {hi}")
            forecast["lower"], forecast["upper"] = np.nan, np.nan
            if lo in predictions:
                forecast["lower"], forecast["upper"] = _values(predictions, lo), _values(predictions, hi)
                if ((forecast.lower > forecast.upper) | (forecast.lower > forecast.predicted)
                        | (forecast.predicted > forecast.upper)).any():
                    raise ValueError(f"intervals must be ordered around {column}")
            table = observed.merge(forecast, on=KEYS, how="left", validate="one_to_one")
            table["model"], table["lead"] = model, lead
            table["issue_date"] = table.valid_date - pd.Timedelta(days=int(lead))
            table["forecast_available"] = table.predicted.notna()
            table["interval_available"] = table[["lower", "upper"]].notna().all(axis=1)
            table["covered"] = table.observed.between(table.lower, table.upper).where(table.interval_available)
            table["observed_high"] = table.observed >= threshold
            table["warned"] = (table.predicted >= threshold).where(table.forecast_available)
            grouped = {key: group for key, group in table.groupby("episode_id", observed=True)}
            for event in catalog.itertuples(index=False):
                scores.append(_score_episode(event, grouped[event.episode_id], model, lead, threshold))
            daily_parts.append(table)
    scores = pd.DataFrame(scores, columns=SCORE_COLUMNS)
    daily = pd.concat(daily_parts, ignore_index=True)
    summary = _summarize(catalog, scores, models, leads, threshold,
                         common_episodes, uncensored_only)
    summary = summary.merge(_daily_summary(daily, models, common_episodes),
                            on=["model", "lead"], validate="one_to_one")
    return EpisodeReport(catalog, scores, summary, daily)


def _summarize(catalog, scores, models, leads, threshold, common_episodes, uncensored_only):
    summaries = []
    censored = catalog.left_censored | catalog.right_censored
    eligible_ids = set(catalog.loc[~censored if uncensored_only else np.ones(len(catalog), bool), "episode_id"])
    scores["scored"] = False
    scores["exclusion_reason"] = np.where(scores.status.eq("complete"), "not_common", "incomplete_forecast")
    excluded_censoring = scores.status.eq("complete") & ~scores.episode_id.isin(eligible_ids)
    scores.loc[excluded_censoring, "exclusion_reason"] = "censored"
    for lead in leads:
        lead_scores = scores.loc[scores.lead.eq(lead)]
        complete = lead_scores.status.eq("complete") & lead_scores.episode_id.isin(eligible_ids)
        counts = lead_scores.loc[complete].groupby("episode_id").size()
        shared = set(counts.index[counts.eq(len(models))])
        for model in models:
            group = lead_scores.loc[lead_scores.model.eq(model)]
            use = group.status.eq("complete") & group.episode_id.isin(eligible_ids)
            if common_episodes:
                use &= group.episode_id.isin(shared)
            selected = group.loc[use]
            scores.loc[selected.index, "scored"] = True
            scores.loc[selected.index, "exclusion_reason"] = "scored"
            detected = selected.loc[selected.detected.eq(True)]
            with_peak_interval = selected.observed_peak_covered.dropna()
            summaries.append({
                "model": model, "lead": lead, "threshold": float(threshold),
                "population": "common_complete" if common_episodes else "model_complete",
                "uncensored_only": uncensored_only, "n_episodes": len(catalog),
                "n_censored": int(censored.sum()), "n_complete": int(group.status.eq("complete").sum()),
                "n_partial": int(group.status.eq("partial").sum()),
                "n_missing": int(group.status.eq("missing").sum()), "n_scored": len(selected),
                "n_excluded": len(group) - len(selected), "n_detected": len(detected),
                "detection_rate": float(selected.detected.mean()) if len(selected) else np.nan,
                "onset_warning_rate": float(selected.onset_warned.mean()) if len(selected) else np.nan,
                "median_peak_ratio": selected.peak_ratio.median(),
                "median_observed_peak_ratio": selected.observed_peak_ratio.median(),
                "mean_peak_bias": selected.peak_bias.mean(),
                "median_abs_peak_timing_days": selected.peak_timing_error_days.abs().median(),
                "n_detected_for_onset": len(detected),
                "median_onset_delay_days": detected.onset_delay_days.median(),
                "n_peak_intervals": len(with_peak_interval),
                "peak_interval_coverage": (float(with_peak_interval.mean())
                                           if len(with_peak_interval) else np.nan),
            })
    return pd.DataFrame(summaries)


def _daily_summary(daily, models, common):
    summaries = []
    daily["scored"] = False
    daily["exclusion_reason"] = np.where(daily.forecast_available, "not_common", "missing_forecast")
    for lead, lead_rows in daily.groupby("lead", sort=True):
        available_counts = lead_rows.loc[lead_rows.forecast_available].groupby(KEYS).size()
        shared = available_counts.index[available_counts.eq(len(models))]
        for model, group in lead_rows.groupby("model", sort=False):
            available = group.forecast_available
            paired = available & (pd.MultiIndex.from_frame(group[KEYS]).isin(shared)
                                  if common else True)
            selected = group.loc[paired]
            daily.loc[selected.index, "scored"] = True
            daily.loc[selected.index, "exclusion_reason"] = "scored"
            warn, high = selected.warned.astype(bool), selected.observed_high
            hits = int((warn & high).sum())
            misses = int((~warn & high).sum())
            false = int((warn & ~high).sum())
            intervals = selected.loc[selected.interval_available]
            summaries.append({
                "model": model, "lead": lead, "n_daily_observed": len(group),
                "n_daily_available": int(available.sum()), "n_daily_missing": int((~available).sum()),
                "n_daily_negative_observations": int(group.observed.lt(0).sum()),
                "n_daily_negative_predictions": int(group.predicted.lt(0).sum()),
                "n_daily_scored": len(selected),
                "n_daily_excluded_common": int(available.sum()) - len(selected),
                "daily_hits": hits, "daily_misses": misses, "daily_false_alarms": false,
                "daily_correct_negatives": int((~warn & ~high).sum()),
                "daily_pod": hits / (hits + misses) if hits + misses else np.nan,
                "daily_far": false / (hits + false) if hits + false else np.nan,
                "daily_csi": hits / (hits + misses + false) if hits + misses + false else np.nan,
                "n_daily_intervals": len(intervals),
                "daily_interval_coverage": float(intervals.covered.mean()) if len(intervals) else np.nan,
                "median_daily_interval_width": (intervals.upper - intervals.lower).median(),
            })
    return pd.DataFrame(summaries)


def plot_episode(report: EpisodeReport, episode_id: str, output: Path) -> None:
    """Draw a saved audit with daily gaps intact and two context days on each side."""
    event = report.catalog.loc[report.catalog.episode_id.eq(episode_id)]
    if len(event) != 1:
        raise ValueError(f"unknown episode ID: {episode_id}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    event = event.iloc[0]
    dates = pd.date_range(event.start_date - pd.Timedelta(days=2),
                          event.end_date + pd.Timedelta(days=2))
    daily = report.daily.loc[report.daily.point_id.eq(event.point_id)]
    leads = sorted(daily.lead.unique())
    models = daily.model.drop_duplicates().tolist()
    threshold = report.summary.threshold.iloc[0]
    fig, axes = plt.subplots(len(leads), 1, figsize=(9, 3 * len(leads)), squeeze=False,
                             sharex=True)
    try:
        for lead, ax in zip(leads, axes[:, 0], strict=True):
            rows = daily.loc[daily.lead.eq(lead)]
            observed = rows.loc[rows.model.eq(models[0])].set_index("valid_date").reindex(dates)
            ax.plot(dates, observed.observed, "o-", color="black", label="Observed", linewidth=2)
            for model in models:
                values = rows.loc[rows.model.eq(model)].set_index("valid_date").reindex(dates)
                line, = ax.plot(dates, values.predicted, "o--", label=model)
                if values.interval_available.eq(True).any():
                    ax.fill_between(dates, values.lower.to_numpy(float), values.upper.to_numpy(float),
                                     color=line.get_color(), alpha=.12)
            ax.axhline(threshold, color="gray", linestyle=":", label="Threshold")
            ax.axvspan(event.start_date - pd.Timedelta(hours=12),
                       event.end_date + pd.Timedelta(hours=12), color="gray", alpha=.08)
            ax.set_title(f"Lead {lead}: {event.point_id}", loc="left")
            ax.set_ylabel("PM2.5 (micrograms/m3)")
            ax.grid(axis="y", alpha=.2)
            ax.legend(fontsize=8, frameon=False, ncols=min(len(models) + 2, 4))
        axes[-1, 0].set_xlabel("Target date (local standard-time day)")
        censored = bool(event.left_censored or event.right_censored)
        fig.suptitle(f"Development episode {episode_id}" + ("; boundary unknown" if censored else ""))
        fig.autofmt_xdate()
        fig.tight_layout()
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=150)
    finally:
        plt.close(fig)


def _development_dates(predictions: pd.DataFrame, years) -> None:
    days = _prediction_dates(predictions)
    if not len(predictions) or not days.year.isin(years).all():
        raise ValueError("issue dates must belong to configured calibration years")
    for column in predictions:
        if match := re.fullmatch(r"y_L(\d+)", str(column)):
            target = days + pd.Timedelta(days=int(match[1]))
            if not target.year.isin(years).all():
                raise ValueError("target dates must belong to configured calibration years")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="development backtest folder or predictions parquet")
    parser.add_argument("--config", type=Path, default=Path("configs/west_us.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--lead", action="append", type=int, dest="leads")
    parser.add_argument("--threshold", type=float, default=USG_THRESHOLD)
    parser.add_argument("--model-populations", action="store_true",
                        help="score each model's complete episodes rather than common episodes")
    parser.add_argument("--uncensored-only", action="store_true")
    parser.add_argument("--plot-episode", action="append", default=[],
                        help="episode ID to plot, repeat for multiple cases; IDs are in catalog.csv")
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH",
                        help="compare one source model across named saved development runs")
    parser.add_argument("--reference-name", default="reference",
                        help="name for the positional run when comparing candidates")
    args = parser.parse_args(argv)
    source = args.run / "predictions.parquet" if args.run.is_dir() else args.run
    try:
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        if (source.parent / "FINAL_RUN_LOCK.json").exists():
            raise ValueError("episode reports accept development runs only")
        predictions, source_digest = _load_predictions(source)
        years = load_study(args.config).raw["split"]["calibrate"]
        _development_dates(predictions, years)
        sources = [(args.reference_name, source, predictions, source_digest)]
        source_model = None
        if args.candidate:
            from serein.eval.episode_runs import combine_runs

            if args.models is not None and len(args.models) != 1:
                raise ValueError("candidate comparison selects exactly one source model")
            for candidate in args.candidate:
                alias, separator, path = candidate.partition("=")
                if not separator or not alias or not path:
                    raise ValueError("candidates must have the form NAME=PATH")
                if alias in {item[0] for item in sources}:
                    raise ValueError("candidate names must be unique")
                path = Path(path)
                path = path / "predictions.parquet" if path.is_dir() else path
                if (path.parent / "FINAL_RUN_LOCK.json").exists():
                    raise ValueError("episode reports accept development runs only")
                table, digest = _load_predictions(path)
                _development_dates(table, years)
                sources.append((alias, path, table, digest))
            source_model = "smokecast" if args.models is None else args.models[0]
            predictions = combine_runs({alias: table for alias, _, table, _ in sources},
                                        model=source_model,
                                        leads=args.leads)
            if args.leads is None:
                pattern = rf"{re.escape(args.reference_name)}_L(\d+)"
                args.leads = sorted(int(match[1]) for column in predictions
                                    if (match := re.fullmatch(pattern, str(column))))
            args.models = [alias for alias, _, _, _ in sources]
        report = analyze_episodes(predictions, models=args.models, leads=args.leads,
                                  threshold=args.threshold, common_episodes=not args.model_populations,
                                  uncensored_only=args.uncensored_only)
        if len(set(args.plot_episode)) != len(args.plot_episode):
            raise ValueError("plot episode IDs must be unique")
        if not set(args.plot_episode).issubset(report.catalog.episode_id):
            raise ValueError("unknown plot episode ID; inspect catalog.csv from an earlier report")
        provenance = []
        for alias, path, table, digest in sources:
            provenance.append({"name": alias, "path": str(path.resolve()), "sha256": digest,
                               "n_input_rows": len(table)})
        metadata = {"source": str(source.resolve()), "sha256": provenance[0]["sha256"],
                    "sources": provenance,
                    "candidate_source_model": source_model,
                    "config": str(args.config.resolve()), "development_years": years,
                    "models": report.summary.model.drop_duplicates().tolist(),
                    "leads": report.summary.lead.drop_duplicates().tolist(), "report_version": 1,
                    "threshold": args.threshold, "common_episodes": not args.model_populations,
                    "uncensored_only": args.uncensored_only,
                    "plots": [{"episode_id": episode_id, "file": f"episode_{index:03d}.png"}
                              for index, episode_id in enumerate(args.plot_episode, start=1)],
                    "interpretation": "exploratory high-PM episodes; no wildfire attribution or health validation"}
        args.output.mkdir(exist_ok=False)
        for name in ("catalog", "scores", "summary", "daily"):
            getattr(report, name).to_csv(args.output / f"{name}.csv", index=False)
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for index, episode_id in enumerate(args.plot_episode, start=1):
            plot_episode(report, episode_id, args.output / f"episode_{index:03d}.png")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.output}: {len(report.catalog)} observed episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
