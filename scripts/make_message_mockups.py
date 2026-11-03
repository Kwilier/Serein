"""Write docs/mockups/messages.md: WhatsApp and SMS versions of the family plan (task E-3).

Uses the real AirPlan code (make_plan, render_sms) with a simulated child and three
made-up forecasts, one per zone. Run:  uv run python scripts/make_message_mockups.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from serein.aqi import CATEGORIES
from serein.personal.action_plan import (
    load_messages,
    make_plan,
    render,
    render_sms,
    sms_segments,
)
from serein.personal.model import ChildModel

OUT = Path(__file__).resolve().parents[1] / "docs" / "mockups" / "messages.md"
ON = date(2026, 8, 18)  # current AQI breakpoints
NAME = "Maya"            # simulated child, not a real person
SCENARIOS = {            # forecast quantiles (µg/m³) for tomorrow
    "green": {0.05: 4.0, 0.25: 6.0, 0.5: 7.0, 0.75: 9.0, 0.95: 12.0},
    "yellow": {0.05: 22.0, 0.25: 33.0, 0.5: 42.0, 0.75: 52.0, 0.95: 70.0},
    "red": {0.05: 95.0, 0.25: 130.0, 0.5: 160.0, 0.75: 190.0, 0.95: 240.0},
}
LANGS = ["en", "es"]


def main() -> None:
    child = ChildModel()  # population prior: a child with no log yet
    lines = [
        "# Message mockups: WhatsApp and SMS (task E-3)",
        "",
        "Made by `scripts/make_message_mockups.py` from the real AirPlan code. The child",
        f"(\"{NAME}\") and the forecasts are made up. **Draft wording:** not yet reviewed",
        "by a clinician (C-8); the Spanish still needs a native-speaker review (C-7). A",
        "third language waits for that review process (C-11).",
        "",
        "## What the channels need",
        "",
        "- **WhatsApp** gets the full plan. Messages a business starts must use a",
        "  template Meta has approved, so each zone (green, yellow, red) and language",
        "  becomes one template with the numbers as fill-in fields.",
        "- **SMS** gets a short version and a link to the full plan. US carriers",
        "  require opt-in and a STOP keyword, so every text ends with one.",
        "- **SMS cost** is per part. Plain text fits 160 characters in one part (153 per",
        "  part when longer). One character outside the GSM-7 set (like á, í, ó, ú, µ)",
        "  switches the whole text to UCS-2: 70 characters, or 67 per part. That is",
        "  why the SMS says \"ug/m3\", and why the Spanish green text says \"Todo",
        "  normal\" instead of \"Día normal\" (3 parts down to 2).",
        "",
        "| Language | Zone | SMS encoding | Characters | Parts billed |",
        "|---|---|---|---|---|",
    ]
    body: list[str] = []
    for lang in LANGS:
        body += ["", f"## {load_messages(lang)['language']}", ""]
        for zone, q in SCENARIOS.items():
            plan = make_plan(child, NAME, q, q[0.5], q[0.5], on=ON, language=lang)
            assert plan.zone == zone, (lang, zone, plan.zone)
            cat = CATEGORIES.index(plan.aqi_category)
            whatsapp = render(plan.zone, plan.reason, NAME, "tomorrow", plan.pm_median,
                              plan.pm_hi, plan.risk, cat, lang, style="whatsapp")
            sms = render_sms(plan.zone, NAME, "tomorrow", plan.pm_median, cat, lang,
                             link="serein.app/p/7Kq2")
            enc, n, parts = sms_segments(sms)
            lines.append(f"| {lang} | {zone} | {enc} | {n} | {parts} |")
            body += [f"### {zone.upper()} (forecast median {plan.pm_median:.0f} µg/m³)", "",
                     "**WhatsApp**", "", "```text", whatsapp, "```", "",
                     f"**SMS** ({enc}, {n} characters, {parts} part{'s' * (parts > 1)})",
                     "", "```text", sms, "```", ""]
    lines += body
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
