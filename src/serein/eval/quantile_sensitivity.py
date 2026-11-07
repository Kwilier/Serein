"""Audit warning decisions under probability brackets implied by forecast quantiles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from serein.aqi import USG_THRESHOLD
from serein.eval.comparison import _prediction_dates
from serein.eval.episode_inputs import _choices, _load_predictions, _values
from serein.eval.intervals import _groups

DEFAULT_LEVELS = (.05, .25, .5, .75, .95)


def quantile_probability_bracket(quantiles, levels, *, threshold: float = USG_THRESHOLD):
    """Bracket P(Y >= threshold) conditional on the supplied quantiles being valid.

    Values are physical concentrations, with the last axis indexing levels.
    These are quantile-implied brackets, not calibrated probability intervals.
    """
    if np.iscomplexobj(quantiles) or np.iscomplexobj(levels):
        raise ValueError("quantiles and levels must be real")
    q, lv = np.asarray(quantiles, dtype=float), np.asarray(levels, dtype=float)
    if (lv.ndim != 1 or not len(lv) or not np.isfinite(lv).all()
            or (lv <= 0).any() or (lv >= 1).any() or (np.diff(lv) <= 0).any()):
        raise ValueError("levels must be finite, strictly increasing probabilities in (0, 1)")
    if q.ndim < 1 or q.shape[-1] != len(lv):
        raise ValueError("quantiles must have a final axis matching levels")
    if (isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold)
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("threshold must be finite and nonnegative")
    if np.isinf(q).any() or (q[np.isfinite(q)] < 0).any():
        raise ValueError("quantiles must be nonnegative and finite or missing")
    # Check every known pair, including pairs separated by missing values.
    prior = np.maximum.accumulate(np.where(np.isfinite(q), q, -np.inf), axis=-1)
    if np.any(np.isfinite(q[..., 1:]) & (q[..., 1:] < prior[..., :-1])):
        raise ValueError("quantiles must be nondecreasing")
    complete = np.isfinite(q).all(axis=-1)
    lower = np.max(np.where(q >= threshold, 1 - lv, 0.), axis=-1)
    upper = np.min(np.where(q < threshold, 1 - lv, 1.), axis=-1)
    # PM concentrations have nonnegative support, so P(Y >= 0) is exactly one.
    if threshold == 0:
        lower = np.ones_like(lower)
        upper = np.ones_like(upper)
    return np.where(complete, lower, np.nan), np.where(complete, upper, np.nan)


def analyze_quantile_sensitivity(
    predictions: pd.DataFrame, *, levels=DEFAULT_LEVELS, models=None, leads=None, groups=(),
    threshold: float = USG_THRESHOLD, cutoffs=(.1, .25, .5, .75, .9),
) -> dict[str, pd.DataFrame]:
    """Report stable/ambiguous fixed-cutoff decisions without changing probabilities.

    Column <model>_q<index>_L<lead> contains a physical concentration at levels[index].
    Optional <model>_prob_L<lead> columns permit comparison with an existing conversion.
    """
    days = _prediction_dates(predictions)
    if predictions.empty or (days.year >= 2024).any():
        raise ValueError("sensitivity requires nonempty development issue dates before 2024")
    levels = np.asarray(levels)
    quantile_probability_bracket(np.empty((0, levels.size)), levels, threshold=threshold)
    cutoffs = list(cutoffs)
    if not cutoffs or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number))
                          or np.iscomplexobj(x) or not np.isfinite(x) or not 0 <= x <= 1
                          for x in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must be unique finite probabilities in [0, 1]")
    cutoffs = sorted(map(float, cutoffs))
    groups = list(groups)
    if len(set(groups)) != len(groups) or any(g not in {"point", "month", "smoke"} for g in groups):
        raise ValueError("groups must be unique choices of point, month and smoke")
    models = _choices(sorted({match[1] for column in predictions
                             if (match := re.fullmatch(r"(.+)_q\d+_L\d+", str(column)))})
                      if models is None else models, "models")
    leads = _choices(sorted({int(match[1]) for column in predictions
                            if (match := re.fullmatch(r".+_q\d+_L(0|[1-9]\d*)", str(column)))})
                     if leads is None else leads, "leads", integers=True)
    if any(((days + pd.Timedelta(days=int(lead))).year >= 2024).any() for lead in leads):
        raise ValueError("sensitivity refuses reserved target dates in 2024 and later")
    masks = list(_groups(predictions, days, groups))
    summaries, audits = [], []
    for lead in leads:
        obs_column = f"y_L{lead}"
        if obs_column not in predictions:
            raise ValueError(f"missing observations: {obs_column}")
        if (not pd.api.types.is_numeric_dtype(predictions[obs_column])
                or pd.api.types.is_complex_dtype(predictions[obs_column])):
            raise ValueError("observations must have a real numeric dtype")
        obs = _values(predictions, obs_column)
        observed, event = np.isfinite(obs), obs >= threshold
        for model in models:
            columns = [f"{model}_q{index}_L{lead}" for index in range(len(levels))]
            missing = set(columns) - set(predictions)
            if missing:
                raise ValueError(f"missing quantile columns: {', '.join(sorted(missing))}")
            probability_column = f"{model}_prob_L{lead}"
            has_probability = probability_column in predictions
            required = [*columns, *([probability_column] if has_probability else [])]
            if any(not pd.api.types.is_numeric_dtype(predictions[column])
                   or pd.api.types.is_complex_dtype(predictions[column]) for column in required):
                raise ValueError("quantiles and probabilities must have real numeric dtypes")
            q = np.column_stack([_values(predictions, column) for column in columns])
            lower, upper = quantile_probability_bracket(q, levels, threshold=threshold)
            complete = np.isfinite(lower)
            p = _values(predictions, probability_column) if has_probability else np.full(len(q), np.nan)
            available = np.isfinite(p)
            if ((p[available] < 0) | (p[available] > 1)).any():
                raise ValueError("supplied probabilities must be in [0, 1] or missing")
            paired = complete & available
            outside = paired & ((p < lower) | (p > upper))
            audit = pd.DataFrame({"point_id": predictions.point_id.to_numpy(), "issue_date": days,
                                  "model": model, "lead": int(lead), "observed": obs,
                                  "has_target": observed, "event": np.where(observed, event, np.nan),
                                  "complete_quantiles": complete, "probability_lo": lower,
                                  "probability_hi": upper, "bracket_width": upper - lower,
                                  "supplied_probability": p, "has_probability": available,
                                  "probability_outside_bracket": outside,
                                  "threshold_below_first": complete & (threshold < q[:, 0]),
                                  "threshold_above_last": complete & (threshold > q[:, -1]),
                                  "threshold_equals_quantile": complete & (q == threshold).any(axis=1)})
            audits.append(audit)
            for group, label, mask in masks:
                selected = mask & complete
                n = int(selected.sum())
                population = {"model": model, "lead": int(lead), "group": group, "group_value": label,
                              "threshold": float(threshold), "n_input": int(mask.sum()),
                              "n_complete": n, "n_incomplete": int((mask & ~complete).sum()),
                              "n_observed": int((selected & observed).sum()),
                              "n_events": int((selected & observed & event).sum()),
                              "n_probability": int((mask & paired).sum()),
                              "n_outside_bracket": int((mask & outside).sum()),
                              "n_threshold_below_first": int((selected & (threshold < q[:, 0])).sum()),
                              "n_threshold_above_last": int((selected & (threshold > q[:, -1])).sum()),
                              "n_threshold_equals_quantile": int((selected & (q == threshold).any(1)).sum()),
                              "mean_bracket_width": float(np.mean(upper[selected] - lower[selected])) if n else np.nan}
                for cutoff in cutoffs:
                    forced_warning = selected & (lower >= cutoff)
                    forced_quiet = selected & (upper < cutoff)
                    ambiguous = selected & ~forced_warning & ~forced_quiet
                    summaries.append({**population, "cutoff": cutoff,
                                      "n_forced_warning": int(forced_warning.sum()),
                                      "n_forced_quiet": int(forced_quiet.sum()),
                                      "n_ambiguous": int(ambiguous.sum()),
                                      "ambiguous_fraction": float(ambiguous.sum() / n) if n else np.nan,
                                      "n_ambiguous_observed": int((ambiguous & observed).sum()),
                                      "n_ambiguous_events": int((ambiguous & observed & event).sum()),
                                      "n_forced_warning_events": int((forced_warning & observed & event).sum()),
                                      "n_forced_quiet_events": int((forced_quiet & observed & event).sum()),
                                      "n_supplied_warnings": int((mask & paired & (p >= cutoff)).sum()),
                                      "n_warning_disagreement": int((mask & paired & (
                                          (forced_warning & (p < cutoff)) | (forced_quiet & (p >= cutoff)))).sum())})
    return {"summary": pd.DataFrame(summaries), "rows": pd.concat(audits, ignore_index=True)}


def write_quantile_sensitivity(source: Path, output: Path, **kwargs) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if (source.parent / "FINAL_RUN_LOCK.json").exists():
        raise ValueError("sensitivity accepts development inputs only")
    predictions, digest = _load_predictions(source)
    result = analyze_quantile_sensitivity(predictions, **kwargs)
    summary = result["summary"]
    manifest = {
        "report_version": 1, "mode": "development_quantile_sensitivity",
        "source": str(source.resolve()), "sha256": digest, "n_input": len(predictions),
        "levels": list(map(float, kwargs.get("levels", DEFAULT_LEVELS))), "units": "micrograms/m3",
        "threshold": float(summary.threshold.iloc[0]), "cutoffs": sorted(summary.cutoff.unique().tolist()),
        "models": summary.model.drop_duplicates().tolist(), "leads": sorted(summary.lead.unique().tolist()),
        "groups": [x for x in summary.group.drop_duplicates() if x != "all"],
        "limitations": [
            "Brackets assume valid quantiles of a nonnegative concentration distribution; they are not calibrated confidence intervals.",
            "CQR coverage does not establish that every reported level is a valid conditional quantile.",
            "Stable decisions are conditional on the bracket assumptions, not a guarantee about real event probabilities.",
            "Equality uses the inclusive event Y >= threshold; repeated quantiles can reveal a conversion's atom convention.",
            "No probabilities, cutoffs or models are fitted, selected or changed.",
            "Missing supplied probabilities are excluded from conversion comparisons and retained in row audits.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    result["summary"].to_csv(output / "summary.csv", index=False)
    result["rows"].to_parquet(output / "rows.parquet", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--lead", action="append", type=int, dest="leads")
    parser.add_argument("--level", action="append", type=float, dest="levels")
    parser.add_argument("--cutoff", action="append", type=float, dest="cutoffs")
    parser.add_argument("--threshold", type=float, default=USG_THRESHOLD)
    parser.add_argument("--group", action="append", choices=("point", "month", "smoke"), dest="groups")
    args = parser.parse_args(argv)
    settings = vars(args).copy()
    source, output = settings.pop("predictions"), settings.pop("out")
    settings["groups"] = settings["groups"] or ()
    for name in ("cutoffs", "levels"):
        if settings[name] is None:
            settings.pop(name)
    try:
        write_quantile_sensitivity(source, output, **settings)
    except (ValueError, TypeError, OSError, KeyError, OverflowError) as exc:
        parser.exit(2, f"quantile sensitivity: {exc}\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
