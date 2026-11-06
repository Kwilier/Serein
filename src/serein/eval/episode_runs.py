"""Align saved candidate runs for paired episode diagnostics."""

from __future__ import annotations

import re

import pandas as pd

from serein.eval.comparison import _prediction_dates
from serein.eval.episode_inputs import _choices, _observations, _values


def combine_runs(runs: dict[str, pd.DataFrame], *, model: str = "smokecast",
                 leads=None) -> pd.DataFrame:
    """Outer-align candidate forecasts, retaining missing rows and target agreement.

    Each run supplies one point model and its optional intervals. Equal observed
    targets are required wherever runs overlap; missing copies can be filled by
    another run. No forecast is filled from another candidate.
    """
    if len(runs) < 2:
        raise ValueError("run comparison requires at least two named runs")
    aliases = _choices(runs, "run names")
    _choices([model], "source model")
    if "y" in aliases:
        raise ValueError("run name y is reserved for observations")
    observations, indexed = {}, {}
    all_leads = set()
    for alias, frame in runs.items():
        days = _prediction_dates(frame)
        _observations(frame, days)
        columns = {int(match[1]): column for column in frame
                   if (match := re.fullmatch(r"y_L(\d+)", str(column)))}
        if len(columns) != sum(bool(re.fullmatch(r"y_L\d+", str(column))) for column in frame):
            raise ValueError("observation lead names must be unambiguous")
        all_leads.update(columns)
        table = frame.copy()
        table["issue_date"] = days
        indexed[alias] = table.set_index(["issue_date", "point_id"])
        observations[alias] = columns
    if leads is None:
        leads = sorted(set.intersection(*(set(columns) for columns in observations.values())))
    leads = _choices(leads, "leads", integers=True)
    keys = indexed[aliases[0]].index
    for alias in aliases[1:]:
        keys = keys.union(indexed[alias].index)
    result = pd.DataFrame(index=keys.sort_values())
    for lead in sorted(all_leads):
        copies = []
        for alias in aliases:
            if lead in observations[alias]:
                copies.append(pd.Series(_values(indexed[alias], observations[alias][lead]),
                                        index=indexed[alias].index).reindex(result.index))
        values = pd.concat(copies, axis=1)
        lo, hi = values.min(axis=1), values.max(axis=1)
        if not lo.eq(hi).where(lo.notna(), True).all():
            raise ValueError(f"candidate runs have different observations at lead {lead}")
        result[f"y_L{lead}"] = lo
    for alias in aliases:
        table = indexed[alias]
        for lead in leads:
            source = f"{model}_L{lead}"
            if lead not in observations[alias] or source not in table:
                raise ValueError(f"{alias} requires observations and {source}")
            for middle in ("", "_lo", "_hi"):
                source, target = f"{model}{middle}_L{lead}", f"{alias}{middle}_L{lead}"
                if source not in table:
                    continue
                if target in result:
                    raise ValueError("run names collide with forecast or interval columns")
                result[target] = pd.Series(_values(table, source), index=table.index).reindex(result.index)
    result = result.reset_index()
    _observations(result, _prediction_dates(result))
    return result
