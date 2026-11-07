# Innovation Brief, questions 6 to 10: sourced draft (UNSENT, team review needed)

Status: draft, 2026-11-07, written from `docs/BUSINESS_NUMBERS_2026-11-07.md` and
`docs/COMPETITOR_VERIFICATION_2026-11-07.md`. Every number has a source in those two
files. Word counts are within the Student Guide limits (counted with the script at
the bottom). `[TEAM: ...]` slots still need a real fact from us. No em dashes, no hype
words, say what we measured.

Replaces the placeholder text for sections 6 to 10 in `innovation_brief.md`. Sections
1 to 5 are unchanged.

---

## 6. Market (300 words) [draft: 298 words]

**Who benefits.** About 4.8 million US children under 18 have current asthma (CDC, 2023 NHIS), including an estimated 548,709 in California, 26,594 in Oregon and 83,411 in Washington (CDC, 2023 NSCH); across the western states, roughly 0.9 to 1.1 million. Each year US children make about 374,000 asthma emergency visits and 61,000 hospital stays (CDC, 2023), at a mean hospital cost of $7,464 per stay (Gabbay et al., 2025). Smoke makes this worse: in San Diego, each 10 µg/m³ of wildfire PM2.5 raised pediatric respiratory emergency and urgent-care visits about 30% (Aguilera et al., 2021); in Washington, asthma emergency visits rose 4% to 12% on smoke days (WA OFM); in Oregon they rose about 25% during the 2020 Labor Day fires (OHA, 2025).

**Who pays.** Three buyer groups already spend money acting on smoke days.

1. School districts: 935 in California, 197 in Oregon, 295 in Washington (1,427; state education agencies, 2026), 13,303 nationally (NCES, 2023-24). Districts already buy safety and messaging tools: one district's ParentSquare contract was $69,913 for 2024-25; Raptor emergency software runs about $2,095 per school building per year.
2. Medicaid managed care plans and children's hospitals: 4.4 million California, 463,000 Oregon and 810,000 Washington children are enrolled in Medicaid or CHIP (KFF, July 2026). Medi-Cal reimburses asthma self-management education at $26.66 per 30-minute unit. In 2025-26 UW Medicine texted smoke guidance to 64,000 asthma and COPD patients using neighborhood-level pollution data; 14% engaged.
3. Air districts and health departments: the Bay Area district alone budgets $1.95 million a year for Spare the Air advertising and alerts.

**Bottom-up size.** At $5,000 per district per year, the 1,427 districts in our three states are a $7.1 million annual market; 70 districts (5%) within three years is $350,000 a year. [TEAM: adjust once the first interviews tell us which districts actually make smoke-day decisions.]

## 7. Competition (300 words) [draft: ~300 words incl. table]

| Who | What they give | What they don't |
|---|---|---|
| EPA AirNow, EnviroFlash | Free official forecast and alerts by "reporting area" (part of a city to a whole county), usually today and tomorrow | Neighborhood detail; live-monitor correction; per-child advice |
| NOAA HRRR-Smoke and AQM (successor: RRFS) | Free model forecasts, 3 to 5 km grid, 48 to 72 hours; AQM has a bias-corrected PM2.5 product | Uncertainty ranges; plain-language advice; any health layer |
| Google Air Quality (BreezoMeter), also inside Apple Weather | 500 m, 96-hour fused forecast from monitors, PurpleAir and smoke models; generic asthma advice | Per-person learning; published uncertainty; head-to-head skill against NOAA or AirNow |
| IQAir AirVisual | 7-day AI forecast, threshold alerts, wildfire notices | Symptom personalization; smoke-model forecasts |
| PurpleAir map | Live sensor readings | Any forecast |
| California Smoke Spotter (CARB) | Next-day statewide smoke forecast app | 2 to 3 day lead; symptom input |
| AirRater (Australia, since 2015) | Per-user symptom-trigger learning with individual alerts; now a district-level forecast | US coverage; child or caregiver mode; Spanish; published accuracy |
| Propeller Health; EPA Smoke Sense | Smart-inhaler tracking; citizen-science symptom diary | Both unavailable: Propeller "has ceased operations" (Resmed); Smoke Sense offline since December 2024 |
| Research: GNN PM2.5 forecasting (Liao et al., 2025); ML correction of HRRR-Smoke (Dong et al.); per-child PM-sensitivity model (Korea, 2025 preprint) | Published methods | A product; uncertainty; a family plan |

**Our difference.** Pieces of Serein exist; the whole does not. Google fuses monitors and smoke models at 500 m but publishes no uncertainty or skill scores and learns nothing about a child. AirRater proved per-user trigger learning works, in Australia, for adults. Serein is the first family-facing US product to join the layers: a monitor-corrected neighborhood forecast with calibrated ranges, scored on held-out seasons against NOAA, AirNow and persistence, turned into a green, yellow or red plan for one child in English or Spanish. Families use it free.

## 8. Go-to-market (150 words) [draft: 124 words]

Start with one partner that already acts on smoke days: a school district or school-nurse group in a smoke-affected part of California, Oregon or Washington. Serein sends its nurses a neighborhood forecast each morning of fire season and gives families a free sign-up link for AirPlan, alongside the AirNow Flag Program many schools already run. Precedent that organizations will do this: UW Medicine reached 64,000 patients with neighborhood smoke guidance in 2025 and renewed for 2026, and Ventura County already texts farmworkers smoke alerts in English and Spanish. First-season success is two numbers: how many smoke days the district was warned about a day ahead, and how many families still log after 30 days. A clinician reviews every message before any family sees it. [TEAM: name any district, nurse, clinic or air district that has agreed to talk; attach letters of support (E-7).]

## 9. Business model (300 words) [draft: 270 words]

Families never pay. Organizations that make decisions on smoke days pay a yearly subscription for the neighborhood forecast, alerts and a dashboard; the personal plan rides along free.

**Pricing, anchored to what buyers already pay.**

- School districts: $500 to $1,000 per school building per year, or about $0.50 to $1.50 per student, with a $1,500 floor. That is a quarter to half of one Raptor emergency-software license ($2,095 per building) and 7% to 21% of one observed ParentSquare contract ($69,913). A 10,000-student district pays $5,000 to $15,000 a year.
- Health plans and children's hospitals: $10 to $20 per enrolled child with asthma per year, less than one Medi-Cal asthma-education unit ($26.66 per 30 minutes). A child asthma hospitalization costs $7,464, so one avoided stay funds 370 to 750 child-years of Serein. We do not yet have evidence that Serein avoids visits; a pilot would measure it.
- Air districts and health departments: a license to embed the forecast in alerts they already send.

**Costs.** All inputs are free public data from NOAA, NASA and EPA. [TEAM: paste the measured median seconds per issue day and the hardware from `outputs/timing/`.] A T4 GPU costs $0.53 an hour on demand, so a daily forecast for the western US costs cents a day, and a small web host is $7 to $25 a month. Text alerts cost about $0.013 per SMS part; our messages fit in one or two parts, so 60 smoke-day alerts to 1,000 families cost about $1,100 a season. WhatsApp utility messages cost $0.0034 each. PurpleAir sensor history, if added, costs about $1 to $5 per sensor-year. At the prices above, compute and messaging together stay under 10% of revenue.

## 10. Fundraising (150 words) [draft: 131 words]

To reach the Summit and run one pilot season we need about $3,000 to $5,000: the $499 Conrad fee (or financial aid), $65 for a provisional patent (micro-entity) and $350 for a trademark application on the name, about $300 for a fire season of hosting, roughly $1,100 of text messages for 1,000 families, $500 for native-speaker and translation review, and travel. Sources: Conrad's prize partners (1517 Fund has given Scholar teams $1,000 plus $10,000 in AWS credits), the Davidson Fellows Scholarship (deadline February 17, 2027), small rolling seed grants such as The Pollination Project, in-kind sensors or data from an air district or PurpleAir, and a crowdfunding page. Federal environmental-justice grants are not realistically available in 2026-27. After the pilot, the first paying district funds the next season. [TEAM: confirm 1517 Fund and AWS are still 2026-27 prize partners.]

---

## Claims elsewhere in the brief to fix before submission (from the competitor check)

1. Never say "first neighborhood-level smoke forecast." Google/BreezoMeter is 500 m, 96 h; NOAA AQM has a bias-corrected PM2.5 product; Trace AQ sells a 4-day ML smoke forecast. Say: first to publish calibrated uncertainty and held-out head-to-head scores in a free family product.
2. Never say "no app personalizes from symptoms." AirRater has since 2015. Say: no US, child-focused, forecast-linked product does.
3. Never say "first per-child smoke-sensitivity model." A 272-child Korean study (2025 preprint) built one. Say: first in a free consumer product using a daily caregiver check-in and a wildfire-smoke forecast.
4. Add a **persistence baseline** to the scored comparison. An MIT paper in BAMS (2026) found that for day-level decisions, morning AirNow persistence matched or beat every operational smoke model. If a judge knows that paper and we have no persistence row, we lose the argument. (The fallback baseline already includes persistence as a last resort; report plain persistence on its own.)
5. HRRR-Smoke's successor RRFS v1 (3 km, 84 h, smoke and dust) was scheduled operational November 3, 2026, with HRRR frozen until about 2028. Say "HRRR-Smoke and its successor RRFS" and plan to ingest RRFS.
6. Propeller Health: Resmed confirms it "has ceased operations" but gives no date. Cite without a date.
7. Smoke Sense: EPA says "unavailable for use" from December 8, 2024 and "down through Summer 2025"; relaunch unknown. Do not say "permanently."
8. Apple Weather: Apple's page says air quality is "provided by BreezoMeter." Do not quote an in-app forecast horizon; it is not documented.
9. Health plans: replace any generic "Kaiser/Blue Shield" mention with the verified UW Medicine 2025-26 outreach (64,000 patients, 14% engaged, 7% booked telehealth).

## Word-count script

```bash
python3 - <<'EOF'
import re
t=open('docs/drafts/innovation_brief_q6-q10_DRAFT.md').read()
for n,h in [(6,'## 6.'),(7,'## 7.'),(8,'## 8.'),(9,'## 9.'),(10,'## 10.')]:
    s=t.index(h); e=t.index('\n## ',s+1) if n<10 else t.index('\n---',s)
    body=t[s:e].split('\n',1)[1]
    body=re.sub(r'\[TEAM:.*?\]','',body,flags=re.S)       # TEAM notes are not submitted
    body=re.sub(r'^\|[-| ]+\|$','',body,flags=re.M)         # table rule lines
    words=re.findall(r"[A-Za-z0-9$%µ³/.,'-]+",body.replace('|',' '))
    print(n,len(words))
EOF
```
