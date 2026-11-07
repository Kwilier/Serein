"""Render pooled fixed-bin development reliability with probability histograms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_reliability(bins: pd.DataFrame, model: str, output: Path) -> None:
    """Keep sparse bins visible and leave unavailable bootstrap bounds unfilled."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    selected = bins.loc[bins.model.eq(model) & bins.group.eq("all") & bins.bin.ge(0)]
    if selected.empty:
        raise ValueError(f"no pooled reliability bins for model {model}")
    if selected.duplicated(["lead", "bin"]).any():
        raise ValueError("plot requires one pooled row per lead and bin")
    figure = Figure(figsize=(11, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for lead, rows in selected.groupby("lead", sort=True):
        rows = rows.sort_values("bin")
        finite = rows.loc[rows.n_scored.gt(0)]
        label = f"Lead {lead}"
        if finite.empty:
            continue
        line, = axes[0].plot(finite.mean_probability, finite.event_rate,
                             marker="o", markersize=4, linewidth=.8, label=label)
        has_bounds = finite[["event_rate_ci_lo", "event_rate_ci_hi"]].notna().all(axis=1)
        axes[0].vlines(finite.loc[has_bounds, "mean_probability"],
                        finite.loc[has_bounds, "event_rate_ci_lo"],
                        finite.loc[has_bounds, "event_rate_ci_hi"], color=line.get_color(), alpha=.6)
        midpoints = (rows.bin_lo + rows.bin_hi) / 2
        axes[1].step(midpoints, rows.n_scored / rows.n_scored.sum(), where="mid",
                     color=line.get_color(), label=label)
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted event probability",
                ylabel="Observed event fraction", title="Pooled reliability")
    axes[1].set(xlim=(0, 1), xlabel="Fixed probability bin midpoint",
                ylabel="Fraction of scored forecasts", title="Scored probability distribution")
    if np.any(selected.n_scored > 0):
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].legend(frameon=False, fontsize=8)
    else:
        axes[0].text(.5, .4, "No scored probabilities", ha="center")
    for ax in axes:
        ax.grid(alpha=.2)
    figure.suptitle(f"Development event probabilities: {model}")
    figure.text(.01, .01,
                "Exploratory calendar-block 95% bounds; no multiplicity adjustment. Empty bins are omitted from the curve.",
                fontsize=8)
    figure.tight_layout(rect=[0, .04, 1, .94])
    with output.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
