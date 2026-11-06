# Comparing saved development forecasts

The backtest already reports paired RMSE differences with day-bootstrap confidence
intervals. A separate tool adds an exploratory Diebold-Mariano (DM) comparison of
daily squared-error losses. It reads `predictions.parquet` without training a model,
downloading data, or changing a backtest run.

```powershell
uv run python -m serein.eval.comparison outputs/backtest/west_us_dev
```

The default writes CSV to the terminal. To save the summary and the daily losses,
choose new paths whose parent directories already exist:

```powershell
uv run python -m serein.eval.comparison outputs/backtest/west_us_dev `
  --output comparison.csv --daily-output comparison_daily.csv
```

Existing files are refused. Neither command runs the reserved evaluation. Use
development exports until the team agrees on the final analysis. This tool does
not freeze the protocol, complete D-5, or establish forecast accuracy on new data.

## What is compared

For each lead and available baseline, the tool keeps rows with finite observations
and finite predictions from both models. Both losses therefore use identical
site-days. Missing values and infinities are excluded from that pair, with counts
in the summary. Different baselines can have different available site-days, so
their results are not a common-population ranking.

Each day's loss difference is the mean across those paired sites of
`(model - observed)^2 - (baseline - observed)^2`. A negative difference favors the
model. The test averages these daily differences with equal weight for each day.
Repeated sites on the same day do not count as independent time observations.
The summary's RMSE columns retain site-day weighting for comparison with the
existing backtest. With unequal numbers of sites per day, the daily and site-day
estimands can differ, including in direction.

`rmse_diff` is the paired site-day RMSE difference. `rmse_diff_ci_lo` and
`rmse_diff_ci_hi` give its percentile 95% interval from the existing whole-day
bootstrap, using 1,000 draws and seed zero by default. `--n-boot` and `--seed`
set those choices explicitly. Whole days are resampled with replacement, retaining
all paired sites on each selected day. This preserves within-day spatial
dependence, but it does not preserve dependence between successive days. At least
five paired days are required for an interval; `bootstrap_status` records when it
is unavailable. A point RMSE difference is still reported for shorter samples.
These intervals are not adjusted for multiple comparisons.

Within each selected subset, the daily CSV retains dates with eligible input rows
even when none have paired observations. Their losses are blank,
their `n_paired` is zero, and they are excluded from inference. Every day's
`n_input`, `n_paired`, and `n_excluded` counts remain available for auditing.
Dates with no subset-eligible rows do not enter that subset's daily audit.
Duplicate `(issue_date, point_id)` keys are rejected before scoring. Issue dates
must be nonmissing, timezone-free calendar dates at midnight. The backtest exports
these as date labels, not forecast timestamps.

Default baselines are `noaa_best`, `aqm_bc`, `hrrr_smoke`, `persistence`,
`climatology`, `gbm`, and `smokecast_nograph`, when present. Leads come from the
`y_L*` observation columns. A selected baseline must have a column at every
selected lead. An incomplete export raises an error instead of silently dropping
a comparison. Quantile bounds are not treated as separate models.

For a comparison chosen in advance:

```powershell
uv run python -m serein.eval.comparison outputs/backtest/west_us_dev `
  --baseline noaa_best --lead 1 --lead 2
```

`--baseline` and `--lead` can be repeated; `--model` changes the first model.
Choose this family before looking at p-values. Selecting a favorable subset after
seeing the full report invalidates its multiple-comparison interpretation.

## Smoke subsets and common populations

The default subset is `all`. To add the study's secondary analysis at monitors
with a known smoke plume overhead yesterday:

```powershell
uv run python -m serein.eval.comparison outputs/backtest/west_us_dev `
  --subset all --subset smoke_yesterday
```

`smoke_yesterday` selects rows with `hms_prev` equal to 1, 2, or 3. Missing HMS
values have unknown eligibility and are excluded from that subset, with their
count in `n_subset_unknown`. They are not relabeled as clear air. Zero means no
plume; other finite codes and infinities are rejected. HMS is only required and
interpreted when the smoke subset is requested.
These counts reflect the saved export. Older exports that encoded missing HMS as
zero need to be regenerated after the missing-data fixes in
[NEXT_STEPS.md](NEXT_STEPS.md).

`n_total` counts all input rows, `n_input` counts rows eligible for that subset,
and `n_subset_excluded` counts the difference. Forecast missingness is then
reported through `n_paired` and `n_excluded` within the subset. A smoke subset
with no eligible rows remains in the summary as an unavailable planned comparison.

For a comparison where all baselines use the same observations at each lead:

```powershell
uv run python -m serein.eval.comparison outputs/backtest/west_us_dev `
  --baseline noaa_best --baseline persistence --common-rows `
  --subset all --subset smoke_yesterday
```

`--common-rows` keeps only rows where the first model, observations, and every
selected baseline are finite at that lead. The intersection is computed separately
for each lead and subset. `population` records this choice;
`n_pairwise_available` shows how many rows the individual pair could have used,
and `n_excluded_common` shows the additional loss from requiring the intersection.
One wholly unavailable baseline makes the common population empty at that lead;
it is not silently removed. Choose the baseline set and population rule before
looking at results. The default `pairwise` rule retains each pair's own finite
rows, matching the backtest's head-to-head comparisons.

## Statistic and calendar gaps

Let `d_i` be a daily mean loss difference, `n` the number of observed days, and
`z_i = d_i - mean(d)`. For maximum lag `q`, the Bartlett covariance sum is:

```text
S = sum_i(z_i^2)
    + 2 * sum_{k=1..q} (1 - k/(q+1)) * sum_{date_i - date_j = k days}(z_i*z_j)
variance(mean(d)) = S/n^2
DM = mean(d) / sqrt(variance(mean(d)))
p_two_sided = 2 * normal_survival(abs(DM))
```

The default `q = max(lead - 1, 0)` covers overlapping daily forecast horizons.
Lead zero and lead one both use lag zero. `--max-lag` sets one calendar-day
bandwidth for all requested leads. A longer bandwidth can be appropriate when
losses remain correlated beyond the forecast horizon; choose and review it on
development data. The default does not establish that longer dependence is absent.

Covariances only use pairs at the specified calendar lag. For example, the last
day of one scoring week and the first day after an omitted week do not become
neighbors. This also prevents covariance pairs across off-season gaps. On a
complete daily calendar, the formula is the usual Bartlett HAC variance. On an
incomplete calendar it is the corresponding covariance sum for the observed
sample mean, with omitted dates contributing no pairs. It does not impute their
losses or correct bias from selectively missing observations.

This is an **asymptotic, two-sided test with a normal reference distribution**.
The Harvey-Leybourne-Newbold small-sample correction is not applied. Its usual
regular-calendar horizon adjustment is not assumed valid for the alternating
scoring weeks. Short samples, persistent smoke episodes, changing site coverage,
and nonstationarity can make the approximation poor. At least
`max(5, q + 2)` observed days are required by this implementation; that guard is
not a guarantee of reliable inference. Insufficient samples and zero estimated
variance produce an explicit status and blank statistics and p-values. Even a
constant nonzero difference does not produce a significance claim.

## Reading the report

`mean_loss_diff` and `dm_stat` are negative when the first model has lower daily
mean squared-error loss. `p_two_sided` tests equal expected daily loss, without
choosing a favorable direction after the fact. `p_two_sided_holm` applies Holm's
step-down correction across all subset/baseline/lead comparisons in this report.
Unavailable planned comparisons still count in the family size, but retain blank
adjusted p-values. This is conservative. No separate correction is made for
repeated reports, searches over settings, or later choices of comparison families.

The CSV records the test `method`, loss weighting, and `family_size`, together
with the bandwidth and bootstrap settings, so saved reports retain these choices.

Report the loss differences, sample counts, unavailable tests, and adjusted
p-values together. A small p-value does not measure practical benefit, validate a
personal health plan, or replace the protocol's primary day-bootstrap RMSE
criterion. The day-level squared-loss estimand and the bootstrap RMSE estimand
must both be described when reporting them.

Treat development-week p-values as exploratory when those outcomes have informed
model settings. Confirmatory use requires a frozen analysis and agreed data
splits. Review model nesting and the dependence assumptions with the team before
choosing the final statistical comparison.

## References and implementation checks

- Diebold and Mariano (1995), *Comparing Predictive Accuracy*, Journal of Business
  & Economic Statistics 13(3), 253-263,
  [doi:10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599).
- The maintained
  [forecast package documentation](https://pkg.robjhyndman.com/forecast/reference/dm.test.html)
  describes Bartlett variance estimation and the modified test. That package's
  small-sample correction is intentionally distinct from this tool's normal
  approximation.
- R's official [p.adjust documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
  describes Holm adjustment and counting unobserved comparisons in the family.
- Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*,
  Scandinavian Journal of Statistics 6(2), 65-70,
  [JSTOR record](https://www.jstor.org/stable/4615733).

The tests compare the lag-zero statistic against an independent one-sample
calculation and the calendar-lag variance against a full Bartlett kernel matrix.
They cover missing weeks, paired missingness, duplicated sites, multiple leads,
Holm adjustment, degenerate losses, and the parquet-to-CSV command. These checks
verify the implementation; they do not establish the statistical assumptions for
the smoke-forecast study. The final analysis still needs team review.
