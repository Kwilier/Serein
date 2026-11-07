# Checking development prediction intervals

The backtest's overall 90% coverage does not establish 90% coverage at each
monitor, in each month, or on smoke days. This offline report separates those
populations and shows both tails of the misses. It reads saved development
predictions without training, recalibrating, downloading inputs, or changing
the source run.

```powershell
uv run python -m serein.eval.intervals outputs/backtest/west_us_dev `
  --output interval_report --group point --group month --group smoke --plot
```

Choose a new output directory whose parent exists. Existing paths are refused.
The report contains `summary.csv`, `daily.parquet`, and `manifest.json`. The
manifest hashes the exact prediction and configuration bytes used, and records
the models, leads, groups, nominal coverage, bootstrap settings and development
years. Only issue and selected target dates in the configuration's calibration
years are accepted. Runs with `FINAL_RUN_LOCK.json` are refused. This command
does not run the reserved evaluation or complete D-5.

`--plot` adds one `coverage_NNN.png` per model, with the model/file mapping in
the manifest. The figure shows pooled and previous-day smoke coverage with
available bootstrap bounds, plus descriptive monitor fractions against their
scored-day counts. Add `--group smoke` and `--group point` to populate those
panels. Unscored populations remain in the tables and are omitted from the
figure. Dotted lines mark the configured nominal coverage. Monthly summaries
remain in the tables. Plotted bootstrap bounds are
not adjusted for the number of groups, and monitor dots are not coverage tests.

## Reading the scores

The input uses the backtest's `issue_date`, `point_id`, `y_L<lead>`,
`<model>_lo_L<lead>` and `<model>_hi_L<lead>` columns. A point prediction is not
needed. By default, interval models and observation leads are discovered from
the export; every selected combination must have both bounds and a target
column. Select explicitly with repeated `--model` and `--lead` options when an
export contains models with different leads. Duplicate site-day keys,
noncalendar issue dates, infinities, and reversed finite bounds are rejected.

Each summary row describes one model, lead and group:

| Column | Interpretation |
|---|---|
| `coverage` | Fraction of scored observations between the bounds, inclusive |
| `lower_miss_rate` | Fraction below the lower bound |
| `upper_miss_rate` | Fraction above the upper bound |
| `mean_width` | Mean upper bound minus lower bound, in micrograms/m3 |
| `interval_score` | Mean width plus a penalty for misses, in micrograms/m3; lower is better |

For nominal coverage `1 - alpha`, the interval score for bounds `lo`, `hi` and
observation `y` is:

```text
hi - lo + (2 / alpha) * max(lo - y, 0) + (2 / alpha) * max(y - hi, 0)
```

The default `alpha` is 0.1 for the backtest's saved 90% bounds. `--alpha` describes
the nominal probability of bounds already in the file. Changing it does not
recompute quantiles or recalibrate a model. Do not relabel saved 90% intervals
as another nominal level to improve their apparent calibration.

Coverage alone rewards very wide intervals. The width and interval score help
expose that tradeoff. A zero-width interval is valid and covers only an exactly
matching observation. The three coverage fractions sum to one when any rows
are scored. A group with no scored rows has blank metrics and
`status=no_scored_rows`, rather than zero coverage.

Scores weight exported site-days equally. `n_input`, `n_observed`, `n_bounds`,
`n_pairwise`, `n_scored`, and `n_excluded` show each denominator. Availability of
observations and bounds is counted separately; those two missingness counts can
overlap. Missing input dates and sites absent from the export cannot be counted.
`daily.parquet` retains input dates with no scored rows and supplies count and
score sums, so the summary can be reconstructed without assuming equal numbers
of sites per day. Divide total sums by total `n_scored`, not by `n_input` or by
the number of days. Daily score sums on dates without scored rows are zero.

## Groups and common populations

An `all` summary is always included. Optional groups are separate partitions of
the exported rows, not intersections:

- `--group point`: one row per monitor ID.
- `--group month`: issue month, including year, rather than target month.
- `--group smoke`: yesterday's HMS code, with 0 labeled `clear`, 1 to 3 labeled
  `smoke`, and missing values labeled `unknown`. Unknown is retained. No other
  finite code or infinity is accepted.

These groups diagnose existing forecasts. HMS identifies a previous-day plume,
not proof that all particulate matter on a target day came from wildfire smoke.
Groups are generated only for labels present in the export. Group totals from
different partitions must not be added together, because they reuse the rows.

By default, each model is scored where its own bounds and the observation are
available. When comparing interval models with different missingness, use
`--common-rows` to require complete bounds from every selected model at each
lead. `n_pairwise` still shows each model's original eligible count, and
`n_excluded_common` shows the additional exclusions. The common population is
specific to the selected lead and group; leads can still have different targets
and missingness.

## Bootstrap limits

The report resamples fixed calendar blocks of issue days. Every site on a day
stays together, and the default `--block-days 7` also keeps nearby issue days
inside a block together. Blocks are numbered by integer division of days since
1970-01-01, so the default seven-day blocks run Thursday through Wednesday.
Missing calendar days stay gaps rather than making distant dates adjacent.
Only blocks with scored rows enter the bootstrap. Each draw samples the same
number of blocks with replacement, then divides the combined count or score
sums by the combined scored-row count. This preserves site-day weighting when
blocks contain different numbers of rows.

`*_ci_lo` and `*_ci_hi` are percentile 95% intervals from 1,000 draws with seed
zero by default. `--n-boot`, `--seed`, and `--block-days` make these choices
explicit. At least five scored blocks are required; otherwise the bounds are
blank and `bootstrap_status=insufficient_blocks`. Five blocks is a reporting
floor, not evidence that asymptotic or bootstrap approximations are reliable.
For an independent-day sensitivity analysis, choose `--block-days 1`.

Blocks are treated as independent. Dependence across their boundaries, long
smoke episodes, seasonal changes, small groups and model selection can make
these intervals misleading. The intervals are exploratory and have no
multiple-comparison adjustment. They do not test a calibrated coverage null or
establish a conditional conformal guarantee. Choose grouping and block length
before comparing candidates, and review sensitivity with the team. Monitor
coverage does not establish uncertainty calibration at new map locations.

## Development example, November 6, 2026

The preserved development export from the rebuilt 2021-22 training and 2023
calibration tables produced the following pooled SmokeCast scores. It contains
five ensemble members, seeds 0 to 4, and was scored on odd ISO weeks of 2023.
The report used the default seven-day blocks, 1,000 draws and seed zero.

| Lead | Scored site-days | Coverage | Exploratory 95% interval |
|---|---|---|---|
| 0 | 25,030 | 0.9036 | 0.8908 to 0.9155 |
| 1 | 25,041 | 0.8904 | 0.8754 to 0.9046 |
| 2 | 24,730 | 0.8983 | 0.8819 to 0.9140 |
| 3 | 24,432 | 0.9096 | 0.8876 to 0.9299 |

All leads had 22 scored calendar blocks. Of 425 exported monitor IDs, 352 had
scored observations at each lead and 73 had none. The point groups retain those
73 unavailable populations. Pooled coverage near 0.9 concealed a monitor with
lead-2 coverage 0.4605 on 76 scored days. A different monitor's lead-0 coverage
was 0.4286 on only seven scored days in two blocks, so the report correctly left
its bootstrap interval unavailable. Check sample counts before comparing groups.

Prediction-file SHA-256:
`7c1751d481ea76e751cc00417befdbde0195d6ab64dd7f07f6b2c9701fc6cf09`.
The summary and daily values agreed between pandas 3.1 and 2.2.3 after normalizing
timestamp storage precision. The source and installed-wheel reports agreed
exactly. These are development diagnostics, with settings selected using 2023;
the underlying legacy NOAA caches lack extraction-registry hashes. They do not
replace the historical results snapshot, verify new-location calibration, or
complete the reserved evaluation.
