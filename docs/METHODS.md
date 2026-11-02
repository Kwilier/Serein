# Methods (plain English first, math second)

## Model 1: SmokeCast

### The problem with today's forecasts
NOAA's HRRR-Smoke simulates where smoke goes using estimated fire emissions and
weather. It's physically smart but often wrong about *how much* smoke reaches
the ground at a given place, and it doesn't look at live sensor readings. NOAA's
AQM is bias-corrected, but only against regulatory monitors and with a slow
statistical correction.

### Our idea
Treat the air sensors as a **network** (a graph). Every sensor is a node; it is
connected to its 10 nearest neighbors within 300 km. A graph neural network (GNN)
lets every sensor "listen" to its neighbors, paying more attention to neighbors
that are **upwind** (our original hypothesis; see the ablation note below). Smoke that is over Redding this morning, with a north wind,
will be over Sacramento tonight.

The GNN doesn't forecast from scratch. It **corrects** NOAA's forecast:

```
forecast = NOAA's AQM forecast  +  learned correction
           (or yesterday's value when AQM doesn't reach that far)
```

So it starts from the physics and only needs to learn where the physics goes
wrong. This is much easier to learn with a few seasons of data.

### Inputs at every node (see `serein/features/table.py`)
- Sensor readings available at issue time: last 1, 3, 6, 24 h, yesterday's mean, 24 h max
- HRRR-Smoke near-surface smoke for each lead day (mean, max)
- HRRR column smoke (smoke aloft that can mix down), wind, boundary-layer height, temperature, humidity
- NOAA AQM bias-corrected PM2.5 for each lead day
- Yesterday's HMS smoke-plume density
- Fire radiative power within 50/100/250 km, and an "upwind fire" index (NOAA HMS fire points; no key needed)
- Location and day of year

### Network
1. Encoder: a small neural net turns each node's inputs into a 96-number summary.
2. Three rounds of message passing. In each round every node gathers messages
   from its neighbors. A neighbor's weight (attention) depends on both nodes'
   summaries, the distance and direction between them, and **how strongly the
   wind blows from the neighbor toward the node**.
3. Output head: for each lead day, 5 quantiles (5/25/50/75/95%) of the correction.

### Training
- Loss: pinball (quantile) loss, so the model learns ranges, not just one number.
- **Node masking:** each step hides 15% of sensors' readings but still asks the
  model to forecast there. This teaches it to forecast where there is *no*
  sensor, which is how we make neighborhood maps.
- Early stopping on even weeks of the calibration season.

### Honest uncertainty: conformal prediction
Neural nets tend to be over-confident. We use **Conformalized Quantile
Regression** (Romano et al., 2019). On calibration days we measure how far
reality fell outside our 90% range, then widen (or narrow) all future ranges by
just enough that 90% of calibration outcomes would have been inside. We do this
separately for clean, moderate, and smoky forecasts, so the range stays honest
on smoke days.

### Math summary
Let x_i be node i's features, h_i^0 = MLP(x_i). For layer ℓ:

    e_ji = [static_ji, w_j·u_ji, w_i·u_ji, |w_j|]        (u_ji = unit vector j→i, w = wind)
    a_ji = softmax_j( MLP_a([h_j, h_i, e_ji]) )           (per head)
    h_i ← LN(h_i + Σ_j a_ji · MLP_m([h_j, h_i, e_ji]));  h_i ← LN(h_i + FFN(h_i))

Output: q_i,L = base_i,L + [m − Σ softplus(·), m, m + Σ softplus(·)] (non-crossing),
in log1p(µg/m³) space. Loss Σ_τ max(τ(y−q_τ), (τ−1)(y−q_τ)).

### Which inputs matter? (ablations, 2026-11-01, dev weeks of 2023)

Each row hides one input group from SmokeCast (3-seed ensembles; `scripts/run_ablations.sh`).
RMSE in µg/m³; percentages are how much worse than the full model.

| Removed | Today | +1 day | +2 days | +3 days |
|---|---|---|---|---|
| nothing (full model) | 4.44 | 7.81 | 9.80 | 11.49 |
| live sensor readings | 7.29 (+64%) | 8.31 (+6%) | 10.18 (+4%) | 11.85 (+3%) |
| HRRR-Smoke + HRRR weather | 5.06 (+14%) | 10.16 (+30%) | 11.30 (+15%) | 11.85 (+3%) |
| NOAA AQM | 5.08 (+14%) | 8.24 (+6%) | 10.73 (+9%) | 12.37 (+8%) |
| HMS smoke plumes | 4.53 (+2%) | 8.12 (+4%) | 9.56 (-2%) | 11.28 (-2%) |
| satellite fire detections | 4.26 (-4%) | 7.84 (0%) | 9.62 (-2%) | 11.28 (-2%) |
| wind direction on graph edges | 4.44 (0%) | 7.77 (-1%) | 9.81 (0%) | 11.49 (0%) |

What this means:
- **No single source is enough.** Live sensors carry "today"; NOAA's physics models
  (HRRR, AQM) carry days 1-3. Serein's value is combining them.
- **Honest negatives.** Fire detections add nothing measurable, likely because HRRR-Smoke
  already uses satellite fire data. Wind direction on graph edges also adds nothing:
  every node already gets the wind as an input, so the network doesn't need it twice.
  Our original "listen to upwind neighbors" idea was not needed; the graph's value
  shows up as forecasting places with no sensor (leave-sites-out test).
- Differences under ~0.2 µg/m³ are within seed-to-seed noise.

## Model 2: AirPlan

### Idea
Every child starts with the same prior belief from published studies; each
logged day updates the belief about *this* child (Bayes' rule). We use logistic
regression for "symptom day yes/no" with a Gaussian prior and the Laplace
approximation (Newton's method with a line search, then the curvature as the
uncertainty). Features: forecast PM2.5, PM2.5 of the 2 previous days,
PM2.5 × time outdoors, colds, missed controller doses.

### From risk to a plan
We compare the child's predicted risk under tomorrow's forecast against the
same child on a clean-air day. The **smoke-caused** extra risk sets the zone.
Uncertainty from both the smoke forecast and the child model is carried through
by Monte Carlo sampling. Public EPA guidance sets a floor: personal plans can
only be stricter, never looser.

### Simulation study (synthetic children, `serein.personal.simulate`)
300 synthetic children, deliberately varied (25% very smoke-sensitive, 50%
typical, 25% not sensitive). Each child gets a simulated 150-day smoke season;
AirPlan learns from the first k days and is scored on the last 50 days, using
smoke *forecasts* with realistic error.

Result (2026-11-01, literature priors, `outputs/airplan/simulation_summary.csv`):
improvement in Brier score over the one-size-fits-all population model:

| days logged | very sensitive | typical | not sensitive |
|---|---|---|---|
| 14 | +2.5% | +2.6% | +2.5% |
| 30 | +5.6% | +3.6% | +3.7% |
| 60 | +8.4% | +4.1% | +4.8% |
| 100 | +10.2% | +5.0% | +6.9% |

After 100 days the 90% credible range of each child's smoke sensitivity contained
the true value 91% (very sensitive) to 98-100% (others) of the time. These are
SIMULATED children: the result shows the method can learn, not that real
children behave this way.

Lesson learned while building: the first version's Newton optimizer could
overshoot on extreme smoke days (PM2.5/10 = 25) and produce absurd estimates. A
line search fixed it (see `tests/test_personal.py`).

### Priors (TODO C-5, 2026-11-01) {#priors}

| Coefficient | Prior (odds ratio) | Evidence | Caveat |
|---|---|---|---|
| Clean-day symptom rate | 12% of days | CDC Asthma Call-back Survey defines *not well controlled* as symptoms on ≥ 9 of 30 days (30%); Illinois 2017 ACBS: 78% of children with asthma well controlled | Our assumption within that range |
| Same-day PM2.5, per +10 µg/m³ | 1.04 (prior SD 0.25 on log scale) | Sugrue et al. 2026, *GeoHealth* 10(2):e2025GH001530: pediatric asthma hospital encounters, wildfire-specific PM2.5, RR 1.04 (0-8%) at lag 0 | Severe events, not daily symptoms; one region (Alameda/Contra Costa) |
| Prior 2 days PM2.5, per +10 µg/m³ | 1.03 (SD 0.20) | Same study: RR 1.02-1.04 at lags 3-4; cumulative RR 1.13 (1.09-1.18) over lags 0-5 | We simplify the lag structure to the 2 previous days (TODO: test longer lags) |
| Extra effect when outdoors | 1.02 (SD 0.15) | No pediatric estimate found | Assumption; outdoor time raises dose |
| Respiratory virus / cold | 4.0 (SD 0.5) | Khetsuriani et al. 2007, *J Allergy Clin Immunol* 119:314: any virus OR 5.6 (2.7-11.6), rhinovirus OR 6.8; a Hong Kong study found OR 2.77 (1.51-5.11) | Exacerbations, not daily symptoms |
| Missed controller medicine | 1.6 (SD 0.4) | Williams et al. 2011, *J Allergy Clin Immunol* 128:1185: HR 0.61 (0.41-0.90) for > 75% vs ≤ 25% ICS adherence; 24% of exacerbations attributable to nonadherence | Mixed-age cohort; some pediatric studies find reverse associations (confounding by severity) |

Evidence we checked and chose **not** to use: Gent et al. 2009 (*Environ Health
Perspect* 117:1168), 149 children in New Haven, found no significant link between
total PM2.5 and daily symptoms, only for traffic-related particle sources (OR 1.10
per 5 µg/m³ for wheeze, 3-day average). An "OR 1.20 per 10 µg/m³" figure for
asthma attacks that turned up in search results could not be verified against
the paper, so it is not used.

Why wide SDs: published numbers are population averages for severe events. The
whole point of AirPlan is that individual children differ, so the prior must let
the log pull a child far from the average.
