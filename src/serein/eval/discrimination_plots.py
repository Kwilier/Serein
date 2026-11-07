"""Plot pooled development event ranking without selecting warning cutoffs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_discrimination(result: dict[str, pd.DataFrame], model: str, output: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    summary = result["summary"]
    selected = summary.loc[summary.model.eq(model) & summary.group.eq("all")]
    if selected.empty or selected.lead.duplicated().any():
        raise ValueError("plot requires one pooled summary per selected model and lead")
    curves = result["curves"]
    curves = curves.loc[curves.model.eq(model) & curves.group.eq("all")]
    if curves.duplicated(["lead", "warn_none", "cutoff"]).any():
        raise ValueError("plot requires unique pooled curve thresholds")
    figure = Figure(figsize=(11, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    plotted = False
    for row in selected.itertuples(index=False):
        if row.ranking_status != "ok":
            continue
        points = curves.loc[curves.lead.eq(row.lead)].sort_values(
            ["warn_none", "cutoff"], ascending=[False, False])
        if points.empty:
            raise ValueError(f"no pooled curve for lead {row.lead}")
        label = f"Lead {row.lead} (events {row.n_events}/{row.n_scored})"
        line, = axes[0].plot(points.false_positive_rate, points.recall,
                             label=f"{label}; AUC {row.roc_auc:.3f}")
        # Precision is undefined for warn-none; use the conventional (0, 1) display endpoint.
        precision = points.precision.where(~points.warn_none, 1.)
        axes[1].step(points.recall, precision, where="pre", color=line.get_color(),
                     label=f"{label}; AP {row.average_precision:.3f}")
        axes[1].axhline(row.event_rate, color=line.get_color(), linestyle=":", linewidth=.8)
        plotted = True
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=.8)
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (dotted: event rate)")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
        if plotted:
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(.5, .5, "Both event classes required", ha="center")
    figure.suptitle(f"Development event ranking: {model}")
    figure.text(.01, .01, "Row-weighted exploratory curves; no rank-score confidence bounds or cutoff selection.",
                fontsize=8)
    figure.tight_layout(rect=[0, .04, 1, .94])
    with output.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
