"""Local calendar figures for development split audits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from serein.eval.comparison import _calendar_dates


def plot_split_audit(assignments: pd.DataFrame, original: pd.DataFrame,
                     proposed: pd.DataFrame, output: Path) -> None:
    """Show issue-day assignments and shared finite labels for every issue year.

    The calendars show original roles; crosses mark proposed exclusions. Lower
    panels show shared-label counts on target dates, not issue dates. A zero
    shared-label count says nothing about availability or statistical independence.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(f"plot already exists: {output}")
    if assignments.empty:
        raise ValueError("plot requires issue assignments")
    days = _calendar_dates(assignments.issue_date)
    years = set(days.year)
    for table in (original, proposed):
        years.update(_calendar_dates(table.target_date).year)
    years = sorted(years)
    roles = list(dict.fromkeys(assignments.role))
    pairs = list(dict.fromkeys(zip(original.role_a, original.role_b)))
    for pair in zip(proposed.role_a, proposed.role_b):
        if pair not in pairs:
            pairs.append(pair)
    fig, axes = plt.subplots(2, len(years), figsize=(5 * len(years), 6),
                             squeeze=False, sharey="row", layout="constrained")
    colors = plt.get_cmap("tab10")
    try:
        for column, year in enumerate(years):
            upper, lower = axes[:, column]
            in_year = days.year == year
            for index, role in enumerate(roles):
                selected = in_year & assignments.role.eq(role).to_numpy()
                kept = selected & assignments.keep_proposed.to_numpy(dtype=bool)
                excluded = selected & ~assignments.keep_proposed.to_numpy(dtype=bool)
                upper.scatter(days[kept], np.full(kept.sum(), index), marker="|",
                              s=180, color=colors(index % 10))
                upper.scatter(days[excluded], np.full(excluded.sum(), index), marker="x",
                              s=18, color="0.5")
            upper.set_yticks(range(len(roles)), roles)
            upper.set_title(str(year))
            upper.set_ylim(-.6, len(roles) - .4)
            for index, (a, b) in enumerate(pairs):
                for table, style, label in ((original, "-", "original"),
                                             (proposed, "--", "proposed")):
                    target = _calendar_dates(table.target_date)
                    subset = table.loc[table.role_a.eq(a) & table.role_b.eq(b)
                                       & (target.year == year)]
                    # Audit days absent from a pair have zero shared labels, not
                    # zero observations. Reindex only this count, never targets.
                    grid = pd.date_range(f"{year}-01-01", f"{year}-12-31")
                    series = pd.Series(subset.n_shared_labels.to_numpy(),
                                       index=pd.DatetimeIndex(subset.target_date))
                    values = series.reindex(grid, fill_value=0)
                    lower.plot(grid, values, style, color=colors(index % 10), linewidth=1,
                               label=f"{a} / {b}: {label}")
            for ax in (upper, lower):
                ax.set_xlim(pd.Timestamp(year=year, month=1, day=1),
                            pd.Timestamp(year=year, month=12, day=31))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                ax.grid(axis="x", alpha=.2)
            lower.set_xlabel("Local target date")
            upper.set_xlabel("Issue date: colored bars retained, gray crosses excluded")
        axes[0, 0].set_ylabel("Original role")
        axes[1, 0].set_ylabel("Shared finite monitor-day labels")
        maximum = max([0] + [float(table.n_shared_labels.max())
                              for table in (original, proposed) if not table.empty])
        axes[1, 0].set_ylim(0, max(1, maximum * 1.05))
        if pairs:
            axes[1, -1].legend(fontsize=7, loc="upper right")
        fig.suptitle("Development split audit: proposed exclusions only, no model refit")
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
