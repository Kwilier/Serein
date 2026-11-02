"""From forecast + personal risk to a family action plan (green / yellow / red).

SAFETY RULE: the personal model may only make advice STRICTER than the public
EPA guidance for children with asthma, never looser. A child who looks
"insensitive" in their log still gets at least the zone the AQI calls for.

Floors from EPA guidance for sensitive groups (children with asthma):
  forecast AQI category >= USG (orange)          -> at least YELLOW
  forecast AQI category >= Very Unhealthy        -> RED

Personal escalation uses the risk CAUSED BY SMOKE: the child's predicted risk
under the forecast vs. the same child, same day, with clean air (5 µg/m³ today
and the two days before). A child whose asthma flares for other reasons (a cold)
should not get a smoke warning on a clean day.
  smoke-attributable: +5 points or 1.3x -> YELLOW;   +15 points or 2x -> RED
These cut-offs are design choices to review with a clinician (TODO C-8).
Serein is not a medical device; every message says to follow the child's own
asthma action plan from their doctor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from serein.aqi import CATEGORIES, pm25_to_category

ZONES = ["green", "yellow", "red"]
MSG_DIR = Path(__file__).parent / "messages"


@dataclass
class Plan:
    zone: str
    reason: str
    risk: float
    risk_lo: float
    risk_hi: float
    pm_median: float
    pm_hi: float
    aqi_category: str
    message: str


def choose_zone(risk: float, baseline_risk: float, pm_median: float, pm_hi: float,
                on: date | None = None) -> tuple[str, str]:
    cat_med = int(pm25_to_category(np.array([pm_median]), on)[0])
    cat_hi = int(pm25_to_category(np.array([pm_hi]), on)[0])
    zone, reason = 0, "personal_low"
    # Public-guidance floor (uses the median forecast; a high upper range nudges up).
    if cat_med >= 2 or cat_hi >= 3:
        zone, reason = 1, "aqi_floor"
    if cat_med >= 4:
        zone, reason = 2, "aqi_floor"
    # Personal escalation on smoke-attributable risk.
    rel = risk / max(baseline_risk, 1e-6)
    excess = risk - baseline_risk
    if (rel >= 1.3 or excess >= 0.05) and zone < 1:
        zone, reason = 1, "personal"
    if (rel >= 2.0 or excess >= 0.15) and zone < 2:
        zone, reason = 2, "personal"
    return ZONES[zone], reason


def load_messages(language: str) -> dict:
    path = MSG_DIR / f"{language}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No messages for '{language}'. Available: "
                                f"{sorted(p.stem for p in MSG_DIR.glob('*.yaml'))}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def render(zone: str, reason: str, child_name: str, day_label: str, pm_median: float,
           pm_hi: float, risk: float, category_idx: int, language: str = "en") -> str:
    msgs = load_messages(language)
    day_label = msgs.get("day_labels", {}).get(day_label, day_label)
    z = msgs["zones"][zone]
    cat = msgs["categories"][CATEGORIES[category_idx]]
    lines = [
        z["title"].format(name=child_name, day=day_label),
        msgs["forecast_line"].format(pm=round(pm_median), pm_hi=round(pm_hi), category=cat),
        msgs["risk_line"].format(risk=round(100 * risk), name=child_name, day=day_label),
    ]
    if reason == "personal":
        lines.append(msgs["personal_note"].format(name=child_name))
    lines.append("")
    lines += [f"- {a.format(name=child_name)}" for a in z["actions"]]
    lines += ["", msgs["emergency"].format(name=child_name), msgs["disclaimer"]]
    return "\n".join(lines)


def make_plan(child_model, child_name: str, pm_quantiles: dict, pm_lag1: float,
              pm_lag2: float, outdoor_frac: float = 0.3, cold: int = 0, missed: int = 0,
              day_label: str = "tomorrow", on: date | None = None,
              language: str = "en") -> Plan:
    r = child_model.risk_under_forecast(pm_quantiles, pm_lag1, pm_lag2, outdoor_frac, cold, missed)
    # Same child, same day, clean air (fixed seed => paired Monte Carlo draws).
    base = child_model.risk_under_forecast({0.05: 5.0, 0.5: 5.0, 0.95: 5.0}, 5.0, 5.0,
                                           outdoor_frac, cold, missed)
    qs = sorted(pm_quantiles)
    med = float(np.interp(0.5, qs, [pm_quantiles[q] for q in qs]))
    hi = float(pm_quantiles[qs[-1]])
    zone, reason = choose_zone(r["mean"], base["mean"], med, hi, on)
    cat = int(pm25_to_category(np.array([med]), on)[0])
    msg = render(zone, reason, child_name, day_label, med, hi, r["mean"], cat, language)
    return Plan(zone, reason, r["mean"], r["lo"], r["hi"], med, hi, CATEGORIES[cat], msg)
