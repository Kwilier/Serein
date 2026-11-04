"""What would Serein change for families? (TODO D-8)

Turns the warning scores into plain numbers: of the days that really turned
orange-or-worse (AQI >= 101, "Unhealthy for Sensitive Groups"), how many would a
family have been warned about the day before, and how many warnings were false
alarms, with Serein vs. EPA's official forecast, on the same reporting-area days.

Scale: 4.8 million U.S. children (6.7%) have asthma (CDC, 2023 NHIS).

    uv run python -m serein.eval.impact outputs/backtest/west_us_dev
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

US_CHILDREN_WITH_ASTHMA = 4_804_000  # CDC, 2023 National Health Interview Survey


def impact(run: Path, lead: int = 1) -> dict:
    e = pd.read_csv(run / "vs_epa_forecast.csv")
    e = e[e["lead"] == lead].set_index("who")
    for who in ("smokecast", "epa_forecast"):
        if (e.index == who).sum() != 1:
            raise ValueError(f"Expected exactly one {who} row for lead {lead}")
    for field in ("usg_events", "n_area_days"):
        if e.loc["smokecast", field] != e.loc["epa_forecast", field]:
            raise ValueError("Impact comparisons require the same reporting-area population "
                             f"for both forecasts ({field} differs)")
    ev = int(e.loc["smokecast", "usg_events"])

    def counts(who):
        pod, far = e.loc[who, "usg_pod"], e.loc[who, "usg_far"]
        hits = pod * ev if ev else 0.0
        if hits == 0 and pd.isna(far):
            warnings = 0.0
        else:
            warnings = hits / (1 - far) if far < 1 else float("nan")
        return {"warned_days": hits, "missed_days": ev - hits, "false_alarms": warnings - hits}

    s, p = counts("smokecast"), counts("epa_forecast")
    return {
        "lead": lead, "smoky_area_days": ev, "n_area_days": int(e.loc["smokecast", "n_area_days"]),
        "serein": s, "epa": p,
        "extra_warned_days": s["warned_days"] - p["warned_days"],
        "fewer_false_alarms": p["false_alarms"] - s["false_alarms"],
        "pct_smoky_days_warned": (
            (100 * s["warned_days"] / ev, 100 * p["warned_days"] / ev)
            if ev else (float("nan"), float("nan"))
        ),
    }


def summary_text(r: dict) -> str:
    s, p = r["serein"], r["epa"]
    a, b = r["pct_smoky_days_warned"]
    ahead = "a day ahead" if r["lead"] == 1 else f"{r['lead']} days ahead"
    percentages = (
        f"caught {s['warned_days']:.0f} of them ({a:.0f}%) vs. {p['warned_days']:.0f} ({b:.0f}%) "
        "for EPA's official forecast"
        if r["smoky_area_days"] else
        "and EPA's official forecast had no smoky days to calculate warning percentages"
    )
    if pd.notna(s["false_alarms"]) and pd.notna(p["false_alarms"]):
        alarms = f"{s['false_alarms']:.0f} false alarms vs. {p['false_alarms']:.0f}."
    else:
        alarms = " ".join(
            f"{name} had {counts['false_alarms']:.0f} false alarms."
            if pd.notna(counts["false_alarms"]) else
            f"{name}'s false-alarm count is unavailable from the reported ratios."
            for name, counts in (("Serein", s), ("EPA", p))
        )
    return (
        f"Across {r['n_area_days']:,} reporting-area days there were {r['smoky_area_days']} days that "
        f"really reached 'Unhealthy for Sensitive Groups' or worse. Warning {ahead}, Serein "
        f"{percentages}: {r['extra_warned_days']:.0f} more smoky days on which a family "
        f"of a child with asthma would have had time to prepare. {alarms} "
        f"(4.8 million U.S. children have asthma; CDC 2023.)"
    )


if __name__ == "__main__":
    run = Path(sys.argv[1])
    for lead in (1, 2):
        r = impact(run, lead)
        print(f"--- lead {lead} ---")
        print(summary_text(r))
