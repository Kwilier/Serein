# Check development event ranking and warning tradeoffs

A small Brier score can accompany poor detection of rare smoke events. Use this
report with the [probability reliability report](WARNING_RELIABILITY.md) to check
whether high probabilities rank events above non-events, and how fixed warning
cutoffs trade missed events against false alarms. The report does not select a
cutoff, fit probabilities, or change a model.

```powershell
uv run python -m serein.eval.discrimination `
  --predictions development_probabilities.parquet `
  --model smokecast --lead 1 --lead 2 --lead 3 `
  --cutoff 0.1 --cutoff 0.25 --cutoff 0.5 --cutoff 0.75 --cutoff 0.9 `
  --group point --group month --group smoke `
  --block-days 7 --n-boot 1000 --seed 0 --plot `
  --out outputs/development_event_review
```

Use a new output directory. Input is a local Parquet file with unique
`point_id`, `issue_date` keys, observed concentrations `y_L<lead>`, and saved
probabilities `<model>_prob_L<lead>`. Follow the
[probability export instructions](WARNING_RELIABILITY.md#export-actual-probabilities):
the backtest's median and outer bounds cannot reconstruct its underlying
exceedance probability. Retain the model, calibration and input provenance.

The default observed event is `y_L<lead> >= 35.5` micrograms/m3. Equality counts
as an event. `--threshold` changes that concentration event; supplied
probabilities must describe the same event. A warning cutoff is a separate
number: for example, `--cutoff 0.25` warns when `p >= 0.25`. The default cutoffs
are 0.1, 0.25, 0.5, 0.75 and 0.9. Repeated `--cutoff` flags replace that list.
They must be unique finite values in `[0, 1]`. Choose them before inspecting
scoring outcomes and record the reason separately.

Selected issue and target dates must precede 2024. A `FINAL_RUN_LOCK.json` next
to the input is refused. This report does not run the reserved evaluation.

## Ranking

`summary.csv` reports ROC AUC and average precision for each model, lead and
group, along with event counts, non-event counts, distinct event days and
non-event days. Scores weight scored site-days equally. Repeated observations
at nearby sites can still be dependent; many rows need not mean many events.

The exact curve has one row per distinct scored probability, processed from
largest to smallest. All tied probabilities enter together. It also has a
`warn_none=True` endpoint with a missing cutoff. A cutoff of 1 is different:
it warns for probabilities equal to 1. The endpoint keeps precision undefined
in the table because it issues no warnings.

ROC AUC is trapezoidal area under recall versus false positive rate. Tied
event/non-event pairs receive half credit. Average precision is
`sum((recall[i] - recall[i-1]) * precision[i])` over distinct probability
cutoffs. It uses recall steps, not trapezoidal precision-recall area. For a
constant forecast with both classes present, AUC is 0.5 and average precision
equals the event rate. Precision-recall comparisons depend on event prevalence;
read the event counts and use a matched population when comparing models.

Both rank scores are withheld unless both observed classes are present.
`ranking_status` is `ok`, `no_events`, `no_nonevents` or `no_scored_rows`.
This guard also withholds the conventional average-precision value of 1 for
an all-event group. The report provides no confidence bounds for rank scores.
Rank scores do not establish probability calibration or forecast skill at a
particular warning cutoff.

The score definitions follow the official scikit-learn references for
[ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
and [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
checked on 2026-11-06. Tests compare tied and untied forecasts against those
library implementations. Requiring both classes for average precision is this
report's support guard, rather than the library's all-event convention.

## Fixed-cutoff scores

`thresholds.csv` retains the confusion counts at each requested cutoff:

| Count | Meaning |
|---|---|
| `tp` | Warned event |
| `fp` | Warned non-event |
| `fn` | Missed event |
| `tn` | Unwarned non-event |

The reported rates are:

| Rate | Formula |
|---|---|
| `recall` | `tp / (tp + fn)` |
| `false_positive_rate` | `fp / (fp + tn)` |
| `precision` | `tp / (tp + fp)` |
| `false_alarm_ratio` | `fp / (tp + fp)` |
| `warning_rate` | `(tp + fp) / (tp + fp + fn + tn)` |
| `csi` | `tp / (tp + fp + fn)` |

A missing denominator produces a missing rate, not zero. In particular, no
warnings makes precision and false alarm ratio undefined. False positive
rate describes non-events warned about; false alarm ratio describes warnings
that were false. They can differ sharply for a rare event.

The calendar-block bootstrap resamples fixed blocks anchored at 1970-01-01.
It retains all scored sites on an issue date, preserves calendar gaps and
computes row-weighted rates from resampled confusion counts. Only blocks with
scored rows are sampled. `n_blocks`, `n_event_blocks` and `n_nonevent_blocks`
show the available support. There must be at least five scored blocks and
both outcome classes for any bounds. Otherwise `bootstrap_status` is
`insufficient_blocks` or `constant_outcome`.

Each rate also has its own `*_bootstrap_status` and `*_n_boot_valid`. If any
resample has a zero denominator for that rate, its status is
`undefined_resamples` and its bounds are missing. Other rates can retain
bounds. Dropping undefined resamples would silently condition on the rare
event or warning being present. Even when all resamples are defined, support
can be weak: review distinct event days/blocks, change block length as a
documented sensitivity analysis, and do not read these guards as adequacy tests.

A rate observed at exactly zero or one also has no bounds, with per-rate
status `boundary_rate` if its resamples are all defined. For example, a sample
with no warnings retains warning rate zero but does not claim a zero-width
interval around zero. Perfect observed recall likewise retains its score of
one without certifying perfect population recall. `*_n_boot_valid` still
counts defined resamples; it does not certify usable bounds.

Bounds are marginal exploratory 95% percentile intervals. They are not adjusted
for multiple cutoffs, leads, models or groups, and are not simultaneous curve
bands. Dependence between blocks, seasonality and previous tuning can remain.
Percentile endpoints do not establish 95% repeated-sample coverage, even when
both classes are present and every resample has a defined rate. Sparse-event
sampling can still give poor coverage without temporal dependence.
Do not choose a cutoff by maximizing these scores on reused development data.

## Missingness and files

Each model uses rows with finite targets and its own finite probabilities by
default. `--common-rows` restricts every selected model to the same available
rows at each lead. Availability is recalculated per lead. `n_input`,
`n_observed`, `n_probability`, `n_pairwise`, `n_excluded_common`, `n_excluded`
and `n_scored` retain the population audit. Missing forecasts are excluded
from confusion counts; they are not treated as no-warning decisions.

IDs must be nonempty strings and dates must be timezone-free calendar dates at
midnight. Numeric observations/probabilities can be missing, but infinities,
complex values and finite probabilities outside `[0, 1]` are rejected, even
when the corresponding target is missing. `--group smoke` requires `hms_prev`
with 0, 1, 2, 3 or missing; missing smoke labels remain an `unknown` group.

| File | Contents |
|---|---|
| `summary.csv` | Rank scores, class support, event prevalence and population counts |
| `curves.parquet` | Every distinct probability cutoff, the warn-none endpoint, confusion counts and rates |
| `thresholds.csv` | Requested fixed cutoffs, confusion counts, rates, block support and bounds |
| `daily.parquet` | Per-day, model, lead, group and fixed-cutoff population/confusion counts |
| `manifest.json` | Hash of decoded source bytes, event threshold, cutoffs, groups and bootstrap settings |
| `discrimination_NNN.png` | With `--plot`, pooled ROC and precision-recall curves for one model |

Daily counts reproduce the fixed-cutoff scores and bootstrap input without
rerunning inference. The source file remains unchanged. Figures show pooled
groups only, omit unsupported rank curves, and display the warn-none endpoint
at precision 1 by convention. That display endpoint does not fill missing
table precision. Precision-recall steps match the average-precision definition;
the dotted horizontal lines show each lead's observed event rate.

Use the [calendar review](DEVELOPMENT_PLANS.md),
[input-support audit](FEATURE_SUPPORT.md) and
[interval diagnostics](INTERVAL_DIAGNOSTICS.md) alongside this report. Earlier
use of development dates remains a limitation. These diagnostics alone do not
establish an adequate evaluation design or new neighborhood forecasting skill.
