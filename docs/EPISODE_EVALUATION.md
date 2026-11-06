# High-PM episode diagnostics

The episode report checks whether a forecast captures a sustained high-PM period,
its onset, and its peak. It reads saved development predictions, without fitting
models, downloading data, or running the reserved evaluation. This supports B-15's
investigation of underpredicted peaks at days 2 and 3. It does not complete B-15.

```powershell
uv run python -m serein.eval.episodes outputs/backtest/west_us_dev `
  --model smokecast --model noaa_best --lead 1 --lead 2 --lead 3 `
  --output outputs/episode_report
```

The output directory must be new and its parent must already exist. The command
accepts only issue and target dates in the selected study's calibration years and refuses a
run with a reserved-evaluation lock file. Use `--config` for a different study.
It never invokes training or evaluation commands. Do not point it at reserved
observations. The in-memory API, `analyze_episodes`, also accepts synthetic tables
for tests and does not load study configuration.

## Episode definition

The default threshold is 35.5 micrograms/m3, inclusive. Each episode is a maximal
sequence of consecutive local calendar dates at one monitor whose observed daily
PM2.5 meets that threshold. These are **high-PM episodes**, not established
wildfire events. The report does not attribute pollution to smoke or use the HMS
indicator to turn missing observations into clean days. `--threshold` changes the
diagnostic threshold; choose it before comparing settings.

Observations come from every `y_L<lead>` column, even when reporting only selected
forecast leads. The target date is `issue_date + lead` calendar days, matching the
backtest's daily targets. Copies of the same monitor/target observation must agree
within relative and absolute tolerances of 1e-6 and must agree on threshold
classification. The smallest existing copy is retained deterministically. Missing
copies do not override a known observation. Conflicting targets fail the report.

A missing observation or absent date breaks an episode. A missing day immediately
before or after an episode marks its corresponding boundary as censored: the
episode may have started earlier or continued later. Censored episodes remain
visible and count toward `n_censored`. `--uncensored-only` excludes them from
episode summary scores while retaining their catalog and forecast audits. This
option does not change the daily warning population.

## Comparing saved training experiments

To compare one source model across saved runs, name each candidate explicitly:

```powershell
uv run python -m serein.eval.episodes outputs/backtest/west_us_peaks_w1 `
  --reference-name w1 --candidate w3=outputs/backtest/west_us_peaks_w3 `
  --candidate w6=outputs/backtest/west_us_peaks_w6 --model smokecast `
  --lead 1 --lead 2 --lead 3 --output outputs/peak_weight_episodes
```

Each label becomes a model name in the report. Candidate comparison accepts one
source model, defaulting to `smokecast`, and retains that model's intervals from
each run. The manifest records every source path, hash, row count, and the source
model name. Candidates must also have development issue dates.

Forecast keys are outer-aligned: a missing row in one run remains missing, even
if another run has a forecast. Observed targets must match exactly where the same
issue/monitor/lead occurs in multiple runs; a known target can fill a missing
copy. Conflicting observations fail rather than comparing different studies as
though they were paired. The resulting episode catalog uses the union of known
targets, and the usual common-complete-episode filter applies to all candidates.
Without `--lead`, only leads with observations in every candidate are scored.

The historical peak-weight files predate later feature and calibration fixes.
Comparing them is a diagnostic for those saved experiments, not evidence about
the current pipeline. Rebuild tables and rerun development separately before
using any new experiment to support forecasting claims.

## Files and denominators

| File | Contents |
|---|---|
| `catalog.csv` | One observed episode per monitor, independent of forecast availability, with start, end, length, peak, and censoring flags. |
| `scores.csv` | Each episode/model/lead, including complete, partial, and entirely missing forecast sequences. |
| `summary.csv` | Episode peak and detection summaries, counts of exclusions, and daily warning contingency tables. |
| `daily.csv` | One row per known monitor/target observation for each selected model and lead, including absent forecasts and optional interval bounds. |
| `manifest.json` | Source path and SHA-256 of the bytes decoded, study calibration years, selected models/leads, threshold, and population options. |

Both forecast audit files include `scored` and `exclusion_reason` columns. These
identify the exact rows used in the summaries. Episode exclusions distinguish
incomplete forecasts, unknown boundaries when that filter is enabled, and loss
of a common population. Daily exclusions distinguish missing forecasts from
otherwise available rows excluded by another model's missing forecast.

To draw a case, repeat the command with a different output directory and add
`--plot-episode '<episode_id>'`, copying the ID from `catalog.csv`. Repeat that
option for multiple cases. The resulting indexed PNG filenames and their episode
IDs are recorded in the manifest. Each plot shows one panel per lead with
observations, forecasts, available intervals, the threshold, and two context days
on each side. Unknown days remain gaps. It plots only the observed-date audit, so
a forecast for a day with no observed target is not displayed.

A **complete** episode has a finite median/point forecast for every observed day
of that episode at the selected lead. An absent issue row counts as a missing
forecast, even when another lead supplies the target observation. Partial and
missing episodes are audited, but never scored as complete misses.

By default, at each lead the summary uses only episodes complete for **every
selected model**. `n_scored` therefore has the same value for those models. Counts
of complete/partial/missing episodes describe each model before this intersection;
`n_excluded` includes the intersection and censoring exclusions. Populations may
still differ across leads. For lead-to-lead comparisons, inspect the episode IDs
in `scores.csv` rather than assuming equal samples. `--model-populations` instead
uses each model's own complete episodes and labels that choice in the output.

Daily warning scores use every observed day, including clean days, rather than
only days within episodes. They use common finite forecast rows across models at
each lead by default. The report separately counts missing forecasts, rows lost
to this intersection, and observed high days. A missing forecast is unavailable,
not a negative warning. Model-specific populations also apply here when requested.

Historical saved files can contain negative measurements, point forecasts, and
interval bounds. Their signed values are retained for reproducibility, counted in
the daily summary, and treated as below a positive episode threshold. Infinity,
duplicate forecast keys, crossed intervals, and timezone-bearing/non-midnight
issue dates are rejected. Rebuild forecasts separately if their data or calibration
needs correcting; this tool does not silently repair them.

## Reading the metrics

- `peak_ratio` is the largest point forecast **within the observed episode**
  divided by the observed episode peak. `observed_peak_ratio` instead uses the
  forecast for the exact observed peak day. Both are retained because a large
  forecast on the wrong day can obscure a missed peak.
- `peak_bias` is forecast episode maximum minus observed episode maximum, in
  micrograms/m3. Peak timing error is the forecast maximum's target date minus
  the observed maximum's date. Ties choose the earliest calendar date.
- `detected` means at least one episode day has a point forecast at or above the
  threshold. `onset_warned` requires that forecast on the first observed episode
  day. Summary detection and onset rates use complete eligible episodes only.
- `onset_delay_days` is the first warned target day minus observed onset. It is
  missing when no episode day is warned. `first_warning_issue_date` subtracts
  the nominal lead from that target day. `warning_notice_days` is observed onset
  minus that issue date; a negative value means the warning was issued after the
  observed onset. These are fixed-lead sequences, not a replay combining leads.
- `interval_coverage` is the fraction of available daily intervals covering that
  day's observation. `observed_peak_covered` checks the interval on the observed
  peak day. `peak_interval_coverage` summarizes that check over eligible episodes
  with a peak-day interval, with `n_peak_intervals` as its denominator. These are
  pointwise intervals, not simultaneous bounds for an episode maximum.
  Daily interval coverage and median interval width use the scored daily warning
  population restricted to available bounds, with `n_daily_intervals` recorded.
- Daily POD is hits / (hits + misses), FAR is false alarms / (hits + false alarms),
  and CSI is hits / (hits + misses + false alarms). Zero denominators stay missing.
  These warnings use the point forecast threshold, not the separately tuned
  probability-based operational warning rule in the backtest.

For example, an observed sequence of 40, 60, 40 with forecasts 20, 45, 50 is
detected, but its onset is missed. Its maximum forecast occurs a day after the
observed peak. The report exposes both timing errors rather than treating this
as a fully successful warning.

The summaries are exploratory, give episodes equal weight, and do not provide
confidence intervals. Neighboring monitors can observe the same regional event,
so episode counts are not independent sample sizes. Changes in observation
availability can also change the catalog. Inspect exclusions and censoring before
using the report to choose a model, and retain development status in any result
description. It does not establish neighborhood skill, health benefit, or the
coverage of forecasts at locations without monitors.
