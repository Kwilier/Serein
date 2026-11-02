# Serein

**Neighborhood-level wildfire smoke forecasts, 1 to 3 days ahead, with a personal risk plan for each child with asthma.**

> *Serein* (n.): fine rain falling from a clear sky after sunset. Clean air arriving when you need it.

Wildfire smoke now drives a large share of the worst air-quality days in the U.S. Children with asthma are hit hardest. Smoke inflames their airways, and an attack can begin a day or two after exposure. Today's public forecasts have three gaps:

1. **They are coarse.** NOAA's smoke model (HRRR-Smoke) runs on a 3 km grid and is often wrong by a lot when the plume arrives. EPA's AQI forecast covers a whole metro "reporting area".
2. **They ignore what sensors are measuring right now.** About 400 EPA monitors in the western U.S. (and tens of thousands of low-cost PurpleAir sensors) see the smoke arrive hours before the next model run, but the forecast a family reads doesn't use those readings. Serein uses the EPA monitors today; adding PurpleAir sensors is next (TODO A-12, A-14).
3. **They give everyone the same advice.** "Unhealthy for sensitive groups" doesn't tell a parent whether *their* child will be fine at 40 µg/m³ or need the inhaler at 20.

Serein closes all three gaps:

| | What it does | How |
|---|---|---|
| **Model 1: SmokeCast** | Corrects NOAA's smoke forecast with live sensor readings and gives a forecast for every neighborhood, with an honest uncertainty range | A graph neural network on the network of air sensors. Each sensor learns from its neighbors (NOAA forecasts, weather and fires are inputs too). Calibrated with conformal prediction. |
| **Model 2: AirPlan** | Learns each child's own smoke sensitivity from a symptom and inhaler log and turns tomorrow's forecast into a personal action plan in the family's own language | A hierarchical Bayesian model. It starts from what the medical literature says about all kids, then personalizes as the log grows. |
| **Proof: Backtest** | Tests SmokeCast against NOAA's and EPA's real forecasts on the same days. The final test is on 2024-25 fire seasons the model never trained or was tuned on (not run yet; current numbers are 2023 development results) | A pre-registered backtest protocol on held-out seasons, with fixed metrics (see [docs/BACKTEST_PROTOCOL.md](docs/BACKTEST_PROTOCOL.md)) |

---

## Start here (new teammate guide)

**Just want to see the results?** You don't need to install anything. Open [docs/results/](docs/results/README.md): the full dev report, the head-to-head vs. NOAA and EPA, and every chart are there.

**Want to run the code?** About 15 minutes to get set up:

1. Install [git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/) (uv installs Python for you).
2. `git clone https://github.com/htvarkali/Serein` then `cd Serein`
3. `uv sync --all-extras` installs everything (Python 3.12, PyTorch, Streamlit...).
4. `uv run pytest` should say all tests passed. If it does, you're set up.
5. Try the parts that need no data: `uv run serein airplan-demo --seed 3 --language es` (a simulated child's action plan in Spanish).

**Want to train and score models?** You need the data. `./scripts/download_all.sh` pulls 5 fire seasons of free public data. It takes about 2-3 hours, downloads roughly 150 GB but keeps only about 2.2 GB on disk (it reads just the parts of each weather file it needs). No API keys needed. A GPU (NVIDIA) makes training much faster but is not required. Then follow [Quick start](#quick-start) steps 2-4.

**Want to help?** Pick a task from the [TODO list](#todo), put your initials on it, and follow the [team rules](#how-to-contribute-team-rules) (branch, test, pull request). Two rules matter most:
- **Never run the final test** (`--final`) on your own. It runs once, as a team, after the protocol is frozen. See [Finishing the proof](#finishing-the-proof-the-final-test-do-this-once).
- **Never commit data, API keys, or any real child's health information.**

**Not a coder? These tasks need people, not code:**

| Task | What to do |
|---|---|
| E-1 | Interview school nurses, parents of kids with asthma, and a pediatrician (5+ total). Notes go in the Brief's "Opportunity" and "Validation" answers. |
| C-7 | A native Spanish speaker reviews `src/serein/personal/messages/es.yaml`. |
| C-8 | A doctor, nurse or respiratory therapist reviews the action-plan wording and the priors table in `docs/METHODS.md`. |
| E-4 to E-7 | Business plan: market size, competitors, who pays, partner letters (school districts, air districts). |
| Pitch | Film the 3-5 min video and finish the website (`website/`). |
| Brief | Edit the draft Innovation Brief in [docs/drafts/innovation_brief.md](docs/drafts/innovation_brief.md) and fill in every `[TEAM: ...]` slot. |
| Conrad portal | Confirm the Lean Canvas was submitted (due Oct 29) and which category we picked. |

---

## How it works

```mermaid
flowchart LR
    subgraph Free public data
        FIRMS[NOAA HMS + NASA<br/>fire detections]
        HMS[NOAA HMS<br/>smoke plume maps]
        PA[PurpleAir<br/>low-cost sensors<br/>(next step)]
        EPA[EPA AirNow<br/>regulatory monitors]
        HRRR[NOAA HRRR-Smoke<br/>+ weather]
        AQM[NOAA AQM<br/>72 h PM2.5]
    end
    PA --> QC[Sensor QC +<br/>EPA correction]
    EPA --> G
    QC --> G[Sensor graph<br/>nodes = sensors,<br/>edges = neighbors + wind]
    FIRMS --> F[Features at<br/>every node]
    HMS --> F
    HRRR --> F
    AQM --> F
    F --> G
    G --> M1[Model 1: SmokeCast GNN<br/>day 1-3 PM2.5 + range]
    M1 --> MAP[Neighborhood map]
    M1 --> M2[Model 2: AirPlan<br/>personal risk]
    LOG[Child's symptom +<br/>inhaler log] --> M2
    M2 --> PLAN[Action plan<br/>in family's language]
```

**The forecast target.** The 24-hour average PM2.5 (µg/m³) for each local day, at each location. This is the quantity the EPA AQI uses (2024 breakpoints: Good ≤ 9.0, Moderate ≤ 35.4, USG ≤ 55.4, Unhealthy ≤ 125.4, Very Unhealthy ≤ 225.4).

**Issue time.** 18:30 UTC (11:30 am Pacific daylight time) each day, using only data that existed at that moment: the 12z HRRR run (on AWS by ~13:55 UTC), the 12z AQM run (on AWS by ~18:03 UTC), and sensor hours ending by 17:00 UTC. We measured those posting times from AWS timestamps. This "no peeking" rule is enforced in code (`serein/data/availability.py`) and tested.

| Lead | What official guidance exists | What Serein uses |
|---|---|---|
| Day 0 (today) | HRRR-Smoke, AQM, EPA forecast | all of it + live sensors |
| Day 1 (tomorrow) | HRRR-Smoke (to 48 h), AQM, EPA forecast | all of it + sensors |
| Day 2 | AQM only (72 h since July 2021; 48 h before) | AQM + sensors + weather |
| Day 3 | **nothing official for smoke** | sensors + fires + weather. This is where Serein adds the most. |

---

## Data sources (all free, all checked on 2026-11-01)

| Source | What | Access | Key needed? |
|---|---|---|---|
| [NOAA HMS Fire](https://www.ospo.noaa.gov/products/land/hms.html) | Daily VIIRS/MODIS/GOES fire detections + fire radiative power (we use VIIRS/MODIS) | `satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Fire_Points/` | No |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/) (optional) | Same satellites, NASA processing | REST API | Free MAP_KEY |
| [NOAA HMS Smoke](https://www.ospo.noaa.gov/products/land/hms.html) | Daily smoke plume polygons (light/medium/heavy) | `satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/` | No |
| [PurpleAir](https://develop.purpleair.com/) | ~30k low-cost PM2.5 sensors | REST API. **Costs points per row**; see note below | Free key, paid points |
| [EPA AirNow](https://docs.airnowapi.org/) | Hourly regulatory PM2.5 + archived official AQI forecasts | `files.airnowtech.org/airnow/YYYY/YYYYMMDD/` | No |
| [NOAA HRRR-Smoke](https://registry.opendata.aws/noaa-hrrr-pds/) | 3 km smoke (`MASSDEN` 8 m, `COLMD` column) + weather, 48 h | AWS `noaa-hrrr-bdp-pds`, byte-range downloads using the `.idx` files | No |
| [NOAA AQM (NAQFC)](https://registry.opendata.aws/noaa-nws-naqfc-pds/) | Official 72 h PM2.5 forecast (raw + bias-corrected) | AWS `noaa-nws-naqfc-pds` | No |

> **PurpleAir cost note.** PurpleAir bills API calls in "points", so pulling years of history for thousands of sensors costs real money. Our approach: (1) the core backtest runs on **free EPA monitors only**, so the headline result doesn't depend on PurpleAir; (2) PurpleAir is added for a smaller set of sensors, using `serein purpleair estimate` before every download; (3) we ask PurpleAir for research/student access (see TODO D-3).

Gotchas we found (details in [docs/DATA.md](docs/DATA.md)): HRRR `MASSDEN` was labeled µg/m³ before 2021-12-21 and kg/m³ after (the loader detects it). AQMv5 (to 2021-07-19) only runs 48 h. AQM's ready-made 24 h files don't use local calendar days, so we average the hourly files ourselves. EPA forecasts are issued *after* our issue time, which favors EPA in the comparison.

---

## Current results (dev mode, 2026-11-01)

Trained on 2021-22 fire seasons, scored on held-out weeks of 2023. **These are
development numbers, not the final test.** The 2024-25 test seasons are locked
until the protocol is frozen (TODO D-4/D-5).

| Error (RMSE, µg/m³) | Today | +1 day | +2 days | +3 days |
|---|---|---|---|---|
| **SmokeCast** | **4.5** | **7.9** | **9.9** | **11.5** |
| Best NOAA forecast | 10.1 | 11.0 | 12.3 | 15.5 |
| Same model, graph turned off | 4.9 | 8.0 | 10.3 | 12.2 |
| **At monitors hidden from SmokeCast** (no sensor) | **7.1** | **8.5** | **10.0** | **11.5** |
| NOAA at those same hidden monitors | 10.5 | 11.1 | 11.9 | 14.1 |

- Beats the best NOAA forecast at every lead (day-bootstrap 95% CIs all below zero), and
  beats NOAA at 88% of 344 individual monitors (day 1).
- **Warnings a day ahead:** of 248 reporting-area days that really turned orange-or-worse,
  Serein warned on 66% vs. 45% for EPA's official forecast, with fewer false alarms
  (79 vs. 109). Day 2: 63% vs. 36%, but more false alarms (63 vs. 38).
- The graph matters most where there is **no sensor**: 18% lower error today and 8% on
  day 3 vs. the same model without neighbors.
- 90% ranges contain the truth 89-91% of the time.
- Weak spots we report: big peaks are still under-forecast at days 2-3; day-2 warnings
  have more false alarms than EPA's.
- **What drives the skill** (ablations): live sensors matter most for today (+64% error
  without them), HRRR-Smoke for days 1-2 (+30% on day 1), AQM for every day. Fire
  detections and wind-on-edges add nothing measurable (details in `docs/METHODS.md`).

Figures: `outputs/backtest/west_us_dev/`. Map: `outputs/maps/map_20230818.png`.

---

## Repo layout

```
Serein/
├── README.md               ← you are here (TODO list at the bottom)
├── configs/
│   ├── west_us.yaml        ← main study: region, seasons, train/calibrate/test split
│   └── northeast_2023.yaml ← out-of-region transfer test (June 2023 Canadian smoke)
├── docs/
│   ├── CONRAD.md             ← what Conrad requires + how judges score
│   ├── BACKTEST_PROTOCOL.md  ← the rules for "proof"; freeze before the final test
│   ├── METHODS.md            ← model math in plain English + cited priors
│   ├── DATA.md / DATA_DICTIONARY.md ← data sources, quirks, every column
│   ├── ETHICS.md             ← child health data, consent, safety
│   ├── results/              ← snapshot of dev results + figures (no data download needed)
│   └── drafts/               ← unsent drafts (Innovation Brief, PurpleAir request)
├── src/serein/
│   ├── data/               ← downloaders + parsers (one file per source)
│   ├── features/           ← sensor graph + feature tables
│   ├── models/             ← baselines, SmokeCast GNN, conformal calibration, maps
│   ├── personal/           ← AirPlan: personal model, action plans, daily check-in
│   ├── eval/               ← metrics, backtest, figures, impact, transfer test
│   └── cli.py              ← `serein ...` command line
├── scripts/                ← download_all, run_dev_suite, run_ablations, build_website ...
├── app/                    ← demo app: map, family plan, daily check-in
├── website/                ← draft of the Conrad team website (static HTML)
├── tests/                  ← pytest; run before every push
├── data/                   ← local data cache, git-ignored (never commit data)
└── outputs/                ← results + figures, git-ignored
```

## Quick start

You need [uv](https://docs.astral.sh/uv/) and git. Everything else installs itself.

```bash
git clone https://github.com/htvarkali/Serein
cd Serein
uv sync --all-extras          # installs Python 3.12 + all packages (PyTorch with CUDA on Win/Linux)
uv run pytest                 # should be all green

# 1) Download the free data (all 5 seasons: ~2-3 h on a fast connection)
./scripts/download_all.sh
# 2) Build feature tables
uv run serein build-table --split train
uv run serein build-table --split calibrate
uv run serein build-table --split test        # builds features only; does not score
# 3) Everything that scores on 2023 dev weeks (main run, hidden-monitor test, ablations, figures)
./scripts/run_dev_suite.sh
# 4) Demos
uv run serein airplan-demo --seed 3 --language es
uv run serein map --date 2023-08-18 --run west_us_dev
uv run streamlit run app/streamlit_app.py
```

No API keys are needed for the main study. Optional keys (NASA FIRMS, PurpleAir) go in a `.env` file at the repo root (never committed); see `.env.example`.

## Finishing the proof: the final test (do this ONCE)

The 2024-25 seasons have never been scored. Doing it in this order is what makes the
result trustworthy to judges:

1. Read `docs/BACKTEST_PROTOCOL.md` as a team. Change its status line to **FROZEN**.
2. Commit and push it. The commit time is your public proof the rules came first.
3. Run the test (about 10 min on an RTX-class GPU):
   ```bash
   uv run serein backtest --final
   uv run serein backtest --final --tag final_holdout --holdout-sites 0.2 --no-gbm
   uv run python -m serein.eval.figures outputs/backtest/west_us_final
   uv run python -m serein.eval.impact outputs/backtest/west_us_final
   ```
4. Run the out-of-region test once:
   ```bash
   uv run serein build-table --config configs/northeast_2023.yaml --split test   # already built locally
   uv run serein transfer --run west_us_final
   ```
5. Report every number, including any where Serein loses. Update the results table
   above, `website/index.html`, and run `RUN=outputs/backtest/west_us_final ./scripts/build_website.sh`.

`--final` writes a lock file; running it again requires `--force` plus a written reason
in the protocol's change log.

## How to contribute (team rules)

1. **Pick a task** from the TODO list below and put your initials next to it in a commit, e.g. `- [ ] (AT) B-4 ...`. That claims it.
2. **Make a branch**: `git checkout -b b4-wind-edges`.
3. **Write a test** in `tests/` for anything that computes a number.
4. `uv run pytest` and `uv run ruff check .` must pass.
5. **Open a pull request**. One other person reviews before merging.
6. When merged, **tick the box** `- [x]` and add the date.
7. Never commit data files, API keys, or any real child's health information.

---

## Team workstreams

Five people, five lanes. Each lane has an owner, but anyone can pick up any task.

| Lane | Owner | Focus |
|---|---|---|
| **A: Data** | _name_ | Downloaders, cleaning, PurpleAir QC, the "no peeking" rule |
| **B: SmokeCast (Model 1)** | _name_ | GNN, features, uncertainty, the neighborhood map |
| **C: AirPlan (Model 2)** | _name_ | Personal model, medical grounding, action plans, translations |
| **D: Proof & Science** | _name_ | Backtest protocol, metrics, figures, writing the Innovation Brief |
| **E: Product & Pitch** | _name_ | App/demo, user interviews (school nurses, parents, doctors), business plan, video |

## Timeline

Official 2026-27 Conrad dates ([Student Guide](https://conrad.spacecenter.org/wp-content/uploads/2026/08/2026-2027-Conrad-Challenge-Student-Guide-Official.pdf), checked 2026-11-01): the Lean Canvas was due **Oct 29, 2026**; the Innovation Stage (Brief + 3-5 min video + website + $499 fee) closes **Jan 7, 2027, 11:59 pm CT**. Our internal deadline is **Jan 1, 2027**. Full requirements and the judges' scoring guide: [docs/CONRAD.md](docs/CONRAD.md).

| Week of | Milestone |
|---|---|
| Nov 2 | Data pipeline works for 1 season; baselines scored (M1) |
| Nov 16 | First GNN beats raw HRRR on validation season (M2) |
| Nov 30 | Backtest protocol frozen; uncertainty calibrated; AirPlan simulation done (M3) |
| Dec 7 | **Run the final held-out test once.** Freeze numbers (M4) |
| Dec 14 | Demo app + action plans in 3+ languages reviewed by native speakers (M5) |
| Dec 21 | Innovation Brief + business plan drafts; expert feedback (M6) |
| Jan 1 | Everything final (M7) |

---

## TODO

Legend: `[ ]` open · `[x]` done · `(XX)` = initials of whoever claimed it · ⭐ = critical path for the headline result.

### A: Data pipeline
- [x] A-1 Repo scaffold, `pyproject.toml`, uv env with CUDA PyTorch *(2026-11-01)*
- [x] A-2 AirNow hourly PM2.5 downloader + parser (free, no key) *(2026-11-01)*
- [x] A-3 AirNow archived **official AQI forecasts** (`reportingarea.dat`) parser, the EPA baseline *(2026-11-01)*
- [x] A-4 NOAA HMS smoke polygons downloader + "smoke density at point" feature *(2026-11-01)*
- [x] A-5 HRRR-Smoke `MASSDEN` + column smoke `COLMD` + weather (wind, PBL, T, RH) at sites via byte-range downloads *(2026-11-01)*
- [x] A-6 NOAA AQM 72 h PM2.5 extraction at sites (raw + bias-corrected) *(2026-11-01)*
- [x] A-7 Fire features (FRP within 50/100/250 km, upwind FRP). Uses **NOAA HMS fire points (no key needed)**; NASA FIRMS code also included if anyone gets a key *(2026-11-01)*
- [x] A-8 PurpleAir client + **point-cost estimator** + EPA correction (Barkjohn 2021 + high-smoke extension) *(2026-11-01; needs key)*
- [x] A-9 "No peeking" availability rules: what data existed at issue time *(2026-11-01)*
- [ ] A-10 ⭐ Full download for all seasons in `configs/west_us.yaml` (EPA + HMS + HRRR + AQM). Big; run overnight
- [x] A-11 AQM archive: v5 (to 2021-07-19, **48 h only**), v6, v7 all use the same file names; loader tries each *(2026-11-01)*
- [ ] A-12 PurpleAir QC: A/B channel agreement, stuck sensors, indoor sensors removed, flag outliers vs. neighbors
- [x] A-13 HRRR weather at sites: 10 m wind u/v, PBL height, RH, T2m *(2026-11-01; precipitation still TODO)*
- [ ] A-14 Pull PurpleAir history for ~300 sensors in 2 test metros (Sacramento, Portland) after cost estimate is approved
- [x] A-15 Data dictionary generated from the real table: `docs/DATA_DICTIONARY.md` (`scripts/make_data_dictionary.py`) *(2026-11-01)*

### B: SmokeCast (Model 1)
- [x] B-1 Baselines: persistence, climatology, raw HRRR-Smoke + background, raw/BC AQM, EPA forecast *(2026-11-01)*
- [x] B-2 Sensor graph builder (k-nearest neighbors, distance + bearing edge features) *(2026-11-01)*
- [x] B-3 GNN v0: wind-aware message passing, residual on top of AQM/HRRR, quantile outputs, lead days 0-3 *(2026-11-01)*
- [x] B-4 Conformal calibration so "90% range" really contains the truth ~90% of the time *(2026-11-01)*
- [x] B-5 Gradient-boosting baseline (no graph) to prove the graph helps (ablation) *(2026-11-01)*
- [x] B-6 ⭐ Trained on 2021-22, tuned on 2023 dev weeks only; settings frozen in BACKTEST_PROTOCOL §6b *(2026-11-01)*
- [x] B-7 Node masking during training (hides readings + site history of 15% of sensors) *(2026-11-01)*
- [x] B-7b Leave-sites-out test (`--holdout-sites 0.2`): graph cuts error at hidden monitors 18% today, 9% day 1 vs. no-graph *(2026-11-01)*
- [ ] B-8 Add PurpleAir nodes; measure how much the low-cost sensors help (ablation)
- [x] B-9a Map v0: `serein map --date ...` runs the GNN on a 0.25° grid of virtual sensors *(2026-11-01)*
- [ ] B-9b Finer grid (1-2 km) around target metros + census-tract averages
- [x] B-10a Graph vs. no-graph + settings sweep (`scripts/sweep_graph.sh`) *(2026-11-01)*
- [x] B-10b Input ablations (`scripts/run_ablations.sh`): sensors, HRRR, AQM matter; fires and wind-on-edges don't (table in `docs/METHODS.md`) *(2026-11-01)*
- [ ] B-16 Ideas from ablations: smoke-transport features that HRRR lacks (e.g. plume age, fire growth rate), or drop fire inputs to simplify
- [ ] B-11 Explainability: which neighbors and features drove a forecast (attention weights, SHAP for GBM)
- [x] B-12 5-seed ensemble (`--seeds 5`, now default) *(2026-11-01)*
- [x] B-13 Smoke days weighted 6x in training: day-1 forecasts now reach ~72% of the true peak (was 59%) *(2026-11-01)*
- [x] B-14 Places without a sensor now get a base forecast from neighbors' readings (was generic clean air); hidden sensors practice this in training *(2026-11-01)*
- [ ] B-15 Peaks still under-predicted at days 2-3. Options: add 2020 (record fires) as training data. Checked 2026-11-01: 2020 HRRR files have NO smoke fields (HRRR-Smoke started Dec 2020), but AQMv5 + HRRR weather exist, and the model handles missing inputs. Or try a heavy-tailed output

### C: AirPlan (Model 2)
- [x] C-1 Hierarchical Bayesian logistic model (population prior → per-child posterior, Laplace update) *(2026-11-01)*
- [x] C-2 Synthetic cohort simulator: shows the model recovers a child's true sensitivity *(2026-11-01)*
- [x] C-3 Action-plan engine: green/yellow/red zones from personal risk, based on CDC/EPA/AAP guidance *(2026-11-01)*
- [x] C-4 Message templates: English + Spanish draft *(2026-11-01; Spanish needs native-speaker review)*
- [x] C-5 Literature review: priors now cite Sugrue 2026 (GeoHealth), Khetsuriani 2007 & Williams 2011 (JACI), CDC ACBS; see `docs/METHODS.md#priors` *(2026-11-01)*
- [ ] C-5b Have a clinician sanity-check the priors (part of C-8)
- [x] C-6 ⭐ Simulation study: 300 synthetic children, literature priors; personal model beats one-size-fits-all from ~2 weeks of logs, +5-10% by day 100 (`outputs/airplan/`) *(2026-11-01)*
- [x] C-7a Spanish terminology aligned with EPA's official Spanish AQI names ("Insalubre para grupos sensibles"...) and CDC asthma terms *(2026-11-01)*
- [ ] C-7 Translations reviewed by **native speakers** (Spanish first; then Chinese, Vietnamese, Tagalog, Hindi/Punjabi by target city census). A human must do this
- [ ] C-8 Review action-plan wording with a pediatric pulmonologist / asthma educator / school nurse. Record their feedback
- [x] C-9 Daily check-in: 6 taps (`serein/personal/log.py`), "Daily check-in" tab in the app, stays on device *(2026-11-01)*
- [ ] C-10 Add confounders: colds/viruses, pollen, temperature, ozone

### D: Proof & Science
- [x] D-1 Metrics: RMSE, MAE, bias, AQI-category accuracy, USG+ hit rate / false alarms (CSI), CRPS, interval coverage *(2026-11-01)*
- [x] D-2 Backtest runner with strict train / calibrate / test split by season; `--final` lock file *(2026-11-01)*
- [x] D-2b Warning rule: warn when P(≥35.5) beats a cut-off tuned on calibration weeks *(2026-11-01)*
- [ ] D-3 PurpleAir research access: draft ready in `docs/drafts/purpleair_data_request.md` (NOT sent; a team member decides whether to send). No free route found: EPA's `AirNowWildfire.csv` only has ~50 temporary smoke monitors
- [ ] D-4 ⭐ Write and **freeze** `docs/BACKTEST_PROTOCOL.md` before looking at test seasons (commit hash = timestamp)
- [ ] D-5 ⭐ Run final test once; bootstrap 95% CIs by day; Diebold-Mariano test vs. each baseline
- [x] D-6a Northeast June 2023 transfer test ready: data downloaded, `serein transfer` command built and checked on West data *(2026-11-01)*
- [ ] D-6b ⭐ Run the transfer test ONCE, right after the final test: `uv run serein build-table --config configs/northeast_2023.yaml --split test` then `uv run serein transfer --run west_us_final`
- [x] D-7 Figures: error vs. lead, warnings vs. EPA, case study, calibration by smoke level, map of improvement by site, peak report (`python -m serein.eval.figures <run>`) *(2026-11-01)*
- [x] D-8 Impact calculator: smoky days warned a day ahead, Serein vs. EPA (`python -m serein.eval.impact <run>`) *(2026-11-01)*

### E: Product & Pitch
- [ ] E-1 Interviews: 5+ parents of kids with asthma, 2+ school nurses, 1+ pediatric doctor. Notes in `docs/interviews/` (no names)
- [x] E-2 Demo app v0 (`uv run streamlit run app/streamlit_app.py`): map + family plan *(2026-11-01)*
- [ ] E-2b Polish the app with feedback from interviews (E-1)
- [ ] E-3 SMS/WhatsApp message mockups in 3 languages
- [ ] E-4 Business model: free for families; paid for school districts / health systems / insurers (Medicaid asthma programs)
- [ ] E-5 Competitor table: AirNow, PurpleAir map, IQAir, Apple/Google weather AQI, Smoke Sense (EPA). What only Serein does
- [ ] E-6 Lean canvas → Innovation Brief, business plan, 3-min pitch video
- [ ] E-7 Partner letter(s) of support (school district, local air district, asthma nonprofit)

---

## Safety and ethics (short version)

- Serein is **not a medical device** and doesn't replace a doctor's asthma action plan. Every plan says so.
- No real child's data is used without written parental consent. Development uses **synthetic children** only. See [docs/ETHICS.md](docs/ETHICS.md).
- Health messages are **never** machine-translated without native-speaker review.
- All data stays local; no personal health data is uploaded anywhere.

## License

TBD by the team (MIT suggested for code). Data sources keep their own terms.
