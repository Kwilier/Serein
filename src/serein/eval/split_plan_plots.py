"""Local figures for declared development calendars and available targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from serein.eval.comparison import _calendar_dates
from serein.eval.split_plan import ROLES


def plot_plan(ranges: pd.DataFrame, assignments: pd.DataFrame,
              availability: pd.DataFrame, output: Path) -> None:
    """Show declared ranges even when a role has no available observation rows."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if output.exists():
        raise FileExistsError(f"plot already exists: {output}")
    starts, ends = _calendar_dates(ranges.start), _calendar_dates(ranges.end)
    years = sorted({year for start, end in zip(starts, ends)
                    for year in range(start.year, end.year + 1)})
    if not years:
        raise ValueError("plot requires declared ranges")
    issue_days = _calendar_dates(assignments.issue_date)
    colors = plt.get_cmap("tab10")
    # File exports must not depend on the desktop's interactive plotting backend.
    fig = Figure(figsize=(max(9, 4 * len(years)), 7), layout="constrained")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, len(years))
    try:
        for col, year in enumerate(years):
            ax = fig.add_subplot(grid[0, col])
            left, right = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(
                year=year + 1, month=1, day=1)
            for index, role in enumerate(ROLES):
                for start, end in zip(starts[ranges.role.eq(role)], ends[ranges.role.eq(role)]):
                    first, last = max(start, left), min(end + pd.Timedelta(days=1), right)
                    if first < last:
                        ax.barh(index, mdates.date2num(last) - mdates.date2num(first),
                                left=mdates.date2num(first), height=.6, alpha=.2,
                                color=colors(index))
                chosen = assignments.role.eq(role).to_numpy() & (issue_days.year == year)
                present = issue_days[chosen].unique()
                ax.scatter(present, np.full(len(present), index), marker="|", s=100,
                           color=colors(index))
            ax.set_xlim(left, right)
            ax.set_ylim(-.6, len(ROLES) - .4)
            ax.set_yticks(range(len(ROLES)), ROLES)
            ax.set_title(str(year))
            ax.set_xlabel("Issue date")
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.grid(axis="x", alpha=.2)
        ax = fig.add_subplot(grid[1, :])
        total = availability.n_rows.to_numpy(dtype=float)
        finite = availability.n_finite_targets.to_numpy(dtype=float)
        fraction = np.divide(finite, total, out=np.zeros_like(finite), where=total > 0)
        x = np.arange(len(availability))
        palette = [colors(ROLES.index(role)) for role in availability.role]
        ax.bar(x, fraction * 100, color=palette)
        for pos in np.flatnonzero(total == 0):
            ax.annotate("no rows", (pos, 0), xytext=(0, 4), textcoords="offset points",
                        rotation=90, ha="center", fontsize=8)
        ax.set_xticks(x, [f"{row.role}\nL{row.lead}" for row in availability.itertuples()],
                      fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Finite target slots (%)")
        ax.set_title("Available EPA rows only; see tables for sample counts and monitor support")
        ax.grid(axis="y", alpha=.2)
        fig.suptitle("Development plan only: pale bands declared, colored marks EPA rows present")
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
