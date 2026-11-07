# Innovation Video: script and storyboard (task S-9)

Target length 4:00 (limit 3 to 5 min). Requirement from the Student Guide: a "tour
of an Innovation Model," and ours is a working prototype (the app, the map, a plan
being generated). Upload unlisted to YouTube or Vimeo AND export MP4 under 800 MB.

Principles
- Lead with the child, not the model. A judge should feel the problem in the first
  30 seconds. The GNN gets 25 seconds, as an answer to "how," not the headline.
- Every number on screen comes from a file in `docs/results/` or
  `outputs/backtest/west_us_final/`. Use dev numbers only until the final test runs,
  and label them "development results, 2023." Replace after D-5.
- Every team member appears and explains their own part on camera. This is the
  single best defense against "did students really build this?"
- Say "development" and "simulated" where true. Judges reward teams that know
  their own limits; the rubric's Practicality criterion asks for exactly that.
- No stock footage of fires. Screen recordings and the team talking are enough.
- Captions on (many judges watch muted). No music under speech.

Scoring lens (Storytelling and Professionalism, 20%): clarity of the opportunity,
investor appeal, consistency of name, logo and colors with the website and brief.

---

## One-sentence pitch (use everywhere: video open, website hero, brief Q1 first line)

> Tomorrow's smoke forecast is for your whole city. Serein tells a parent what it
> means for their street and their child, a day earlier than the official forecast.

Alternate for a parent audience:

> When wildfire smoke is coming, Serein texts you the night before: how bad it will
> be on your block, and what to do for your kid.

---

## Storyboard

| Time | On screen | Voice (who) | Notes |
|---|---|---|---|
| 0:00-0:25 | Black, then a phone lock screen. A text arrives: the real YELLOW message for "Maya" from `docs/mockups/messages.md`. Cut to a team member on camera. | **[Member A]**: "This text went out at 6 p.m. the night before a smoke day. It says how bad the air will be on one family's block tomorrow, and what to do for their daughter, who has asthma. Four point eight million American kids have asthma. On smoke days, a lot of them end up in the ER. We built Serein so that text exists." | Hook. Say "4.8 million" (CDC 2023). No logo yet. |
| 0:25-0:55 | Side-by-side: EPA AirNow forecast page for a metro "reporting area" vs. a zoomed Serein neighborhood map (`serein map --date 2023-08-18`). | **[A]**: "Today a parent gets one forecast for the whole metro area, usually the same morning. Smoke doesn't work that way; it hits one neighborhood hours before another. And the advice is the same for every child: 'unhealthy for sensitive groups.' That doesn't tell you whether your kid is fine at 40 micrograms or needs the inhaler at 20." | The three gaps, in plain words. Show, don't list. |
| 0:55-1:15 | Title card: Serein logo, tagline. Then four quick face shots. | **[A]**: "We're four students in [city]. [B] built the data pipeline, [C] the forecast model, [D] the personal plan, and I did validation and the pitch." | 20 seconds. Names, grades on screen. Matches brief Q2 and website Team. |
| 1:15-2:00 | Screen recording: Streamlit app, map tab. Hover a neighborhood, show the forecast and the range "about 75, could reach 140." Then a simple diagram: NOAA forecast + live monitors → Serein → neighborhood map. | **[C]**: "Here's how the forecast works. NOAA already runs a smoke model, and about 400 EPA air monitors in the West are measuring smoke right now. The official forecast a family reads doesn't use those live readings. Serein does. Our model treats the monitors as a network: each one learns from its neighbors, plus NOAA's forecast, the weather and where the fires are, and it corrects NOAA's number for every neighborhood, one to three days out. Every forecast comes with a range, and we checked that the real value lands inside our 90% range about 90% of the time." | The only technical minute. Say "graph neural network" once, on screen as a caption, not aloud more than once. |
| 2:00-2:35 | Chart: error vs. lead day, Serein vs. NOAA fallback vs. EPA (from `docs/results/figures/`). Then the warning bar: 71% vs 46%. | **[A]**: "We tested it the way a scientist would: train on 2021 and 2022, then score on fire-season days the model never saw. [Pause.] One day ahead, Serein warned on 71% of the days that really turned unhealthy for sensitive groups. EPA's official forecast warned on 46%, and we raised fewer false alarms. Two days ahead we still catch more days, but with more false alarms than EPA, and we report that too." | **Replace with final 2024-25 numbers after D-5.** Caption: "Development results, 2023 held-out weeks" until then. Showing a weakness on purpose is a feature. |
| 2:35-3:15 | Screen recording: `uv run serein airplan-demo --seed 3 --language es`, then English. A green plan and a red plan appear. Then the 6-tap check-in screen. | **[D]**: "The forecast is only useful if it becomes a decision for one child. Each evening a parent answers six taps: symptoms, rescue inhaler, activity, sick today. Over a few weeks Serein learns how much smoke affects that child, starting from what the medical literature says about kids in general. Then it turns tomorrow's forecast into a green, yellow or red plan, in English or Spanish. Two rules we never break: the plan can only be stricter than EPA's public guidance, never looser, and it always says to follow the child's own doctor's action plan." | If a clinician has reviewed by filming day, add: "Reviewed by [role, affiliation]." Say "simulated" if the child shown is synthetic. |
| 3:15-3:40 | Simple slide: who pays. Icons for school district, health plan, air district. One line of pricing. | **[B]**: "Families never pay. School districts already pay about two thousand dollars a building for emergency software; we price below that. Health plans pay less per child than one asthma-education visit. All our input data is free from NOAA, NASA and EPA, so a forecast for the whole western U.S. costs cents a day." | Numbers from `docs/BUSINESS_NUMBERS_2026-11-07.md`. |
| 3:40-4:00 | Team on camera together. End card: name, tagline, website URL, "Conrad Challenge 2026-27, Health and Nutrition." | **[A]**: "This spring we want to run Serein for one school district through one fire season, with a clinician reviewing every message, and measure two numbers: how many smoke days we warned about a day ahead, and how many families kept logging after a month. Smoke is getting worse. Kids with asthma shouldn't find out from a cough." | Confirm the category once S-1 is done. |

---

## Shot list and assets

- Phone lock-screen text: film a real phone receiving the mockup text (send it to yourself via any SMS app), or screen-record a phone mockup. Use the exact wording from `docs/mockups/messages.md`.
- AirNow reporting-area screenshot: airnow.gov for your metro, forecast tab.
- Serein map: `uv run serein map --date 2023-08-18 --run west_us_dev` (or the final run).
- App: `uv run streamlit run app/streamlit_app.py`, record at 1920x1080, hide the browser chrome.
- Charts: `docs/results/figures/` (rmse-by-lead, warnings vs EPA). Re-export after the final test.
- Plan generation: terminal recording of `airplan-demo`; increase the font size.
- Diagram: three boxes, drawn in Keynote/Slides in the website colors. The mermaid chart in the README is too dense for video.
- Team shots: one location, natural light, phone camera at eye level, lapel mic or quiet room.

## Editing

- Any editor is fine (iMovie, DaVinci Resolve free, CapCut desktop). Export H.264 1080p, under 800 MB (a 4-minute 1080p export is typically 200-400 MB).
- Captions: auto-generate in YouTube, then correct every number by hand.
- Last check before upload: every number on screen matches a file in `docs/results/` or `outputs/backtest/west_us_final/`; the category matches the portal; the website URL works.

## Website hero (S-8) should match

Hero line: the one-sentence pitch above. Sub-line: "A free neighborhood smoke forecast and a personal plan for every child with asthma. Built by students, scored against NOAA and EPA." Three buttons: See the results · Try the demo · Read the safety rules.
