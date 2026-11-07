"""Render descriptive coverage diagnostics from an interval report summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_coverage(summary: pd.DataFrame, model: str, output: Path) -> None:
    """Show pooled/smoke coverage and the available monitor distributions.

    Monitor dots are descriptive fractions, not independent coverage tests.
    Missing bootstrap bounds retain their point estimate without an error bar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    selected = summary.loc[summary.model.eq(model)]
    if selected.empty:
        raise ValueError(f"no interval summaries for model {model}")
    nominal = selected.nominal_coverage.unique()
    if len(nominal) != 1:
        raise ValueError("plot requires one nominal coverage level")
    leads = sorted(selected.lead.unique())
    pooled = selected.loc[selected.group.isin(["all", "smoke"])]
    groups = list(pooled.groupby(["group", "group_value"], sort=True))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    try:
        offsets = np.linspace(-.25, .25, len(groups)) if len(groups) > 1 else [0.]
        for offset, ((group, label), rows) in zip(offsets, groups, strict=True):
            rows = rows.sort_values("lead")
            known = rows.loc[rows.coverage.notna()]
            x = np.array([leads.index(lead) for lead in known.lead]) + offset
            dots = axes[0].scatter(x, known.coverage, s=20,
                                    label="all" if group == "all" else str(label))
            color = dots.get_facecolor()[0]
            has_ci = known[["coverage_ci_lo", "coverage_ci_hi"]].notna().all(axis=1).to_numpy()
            lo, hi = (known.loc[has_ci, column].to_numpy()
                      for column in ("coverage_ci_lo", "coverage_ci_hi"))
            axes[0].vlines(x[has_ci], lo, hi, color=color)
            for bound in (lo, hi):
                axes[0].plot(x[has_ci], bound, linestyle="none", marker="_", color=color)
        axes[0].set(xticks=np.arange(len(leads)), xticklabels=leads,
                    xlabel="Forecast lead (local days)", title="Pooled and previous-day smoke groups")
        axes[0].set_ylabel("Fraction covered")
        if groups:
            axes[0].legend(frameon=False, fontsize=8)
        points = selected.loc[selected.group.eq("point") & selected.n_scored.gt(0)]
        for lead, rows in points.groupby("lead", sort=True):
            axes[1].scatter(rows.n_scored, rows.coverage, s=12, alpha=.4, label=f"Lead {lead}")
        if points.empty:
            axes[1].text(.5, .5, "No scored monitor groups\nUse --group point to include them",
                         ha="center", va="center", transform=axes[1].transAxes, fontsize=9)
        else:
            axes[1].legend(frameon=False, fontsize=8)
        axes[1].set(xlabel="Scored issue days at monitor", title="Monitor fractions (descriptive)")
        for ax in axes:
            ax.axhline(nominal[0], color="black", linestyle=":", linewidth=1)
            ax.set_ylim(0, 1.03)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(f"Development interval coverage: {model}")
        fig.text(.01, .01,
                 "Exploratory 95% bootstrap bounds. Missing estimates are omitted; no multiplicity adjustment.",
                 fontsize=8)
        fig.tight_layout(rect=[0, .04, 1, .94])
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=150)
    finally:
        plt.close(fig)
