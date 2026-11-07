# Check development warning probabilities

Warning hit rates and false alarms depend on a chosen warning cutoff. They do
not establish that a predicted 20% event chance corresponds to events on about
20% of those days. This report scores saved event probabilities with the Brier
score and fixed-bin reliability summaries. It does not fit probabilities or
choose warning cutoffs.

```powershell
uv run python -m serein.eval.reliability `
  --predictions development_probabilities.parquet `
  --group point --group month --group smoke `
  --n-bins 10 --block-days 7 --n-boot 1000 --seed 0 --plot `
  --out outputs/development_probability_review
```

Use a new output directory. The command reads one local Parquet file, records
the hash of the bytes it decoded, and leaves the file unchanged. Issue dates
and selected target dates must precede 2024. A `FINAL_RUN_LOCK.json` next to the
input is refused. This command neither opens nor runs the reserved evaluation.

## Export actual probabilities

The input must have unique `point_id`, `issue_date` keys, observed concentrations
in `y_L<lead>`, and probabilities in `<model>_prob_L<lead>`. For example,
`smokecast_prob_L1` describes the event `y_L1 >= 35.5`, with concentrations in
micrograms/m3. Equality counts as an event. Use repeated `--model` and `--lead`
options to choose a subset. Otherwise models and target leads are discovered
from the columns, and each selected model/lead must have both required columns.
`--group smoke` additionally requires `hms_prev` with 0, 1, 2, 3 or missing.

The standard backtest currently saves medians and outer bounds, not event
probabilities. Those three values cannot reconstruct the full interpolated
probability used by its warning rule. Save probabilities alongside the keyed
rows during a controlled development inference, using all the calibrated
quantiles and the same conversion as the warning rule. For example, given
`q_calibrated` in log1p concentration units with shape `[rows, leads, levels]`:

```python
from serein.eval.backtest import prob_exceed

# rows must be in exactly the same order as q_calibrated.
export = rows[["point_id", "issue_date", *[f"y_L{x}" for x in leads]]].copy()
for index, lead in enumerate(leads):
    export[f"smokecast_prob_L{lead}"] = prob_exceed(
        q_calibrated[:, index], levels, thr=35.5,
    )
export.to_parquet("development_probabilities.parquet", index=False)
```

Retain the model, calibration, input and threshold provenance with that export.
Add `hms_prev` if smoke groups are needed. Do not replace the original backtest
file. Do not label a binary warning decision as the model's underlying
probability. Binary probabilities can be scored, but they describe a different
forecast. The report cannot verify what event a supplied probability represents.
If using `--threshold`, export probabilities for that same concentration event;
changing the flag alone only changes the observed labels.

IDs must be nonempty strings, dates must be timezone-free calendar dates, and
selected observations and probabilities must have real numeric dtypes. Missing
values are retained in audits. Infinities and probabilities outside `[0, 1]`
are rejected even if the corresponding observation is missing.

## Scores and populations

The raw Brier score is `mean((p - event)^2)` on scored rows. It ranges from zero
for perfect probabilities to one for confidently wrong forecasts. It is a
proper probability score, but a low score on rare events can still come from
forecasting a small probability on every day. Read the event frequency and the
reliability curve with the score. No climatology baseline or skill score is
fitted to scoring outcomes.

By default each model uses its own finite probability/observation pairs.
`--common-rows` requires finite probabilities from every selected model for a
lead, so all models use the same site-days at that lead. Common populations can
still differ between leads. `n_pairwise`, `n_scored`, `n_excluded_common` and
the missingness counts make the population change explicit. Rows are equally
weighted; a monitor with more available dates contributes more weight.

`probability_bias` is mean predicted probability minus event fraction. It can
be zero while errors in different probability bins cancel. `ece` is the
scored-row-weighted absolute difference between bin-mean predicted probability
and bin event frequency. It depends on the fixed binning and sample size, and
is not a proper score or a significance test.

Bins have equal widths on `[0, 1]`, fixed before scoring. Interior edges belong
to the bin on their right, and probability one belongs to the last bin. Empty
bins remain in the table with zero counts and missing rates. Bin `-1` contains
missing probabilities and is retained for the input audit. It never contains
scored forecasts. A valid probability without an observed target remains in
its probability bin but contributes no score.

The binned Brier decomposition is:

```text
reliability = sum(weight_bin * (mean_probability_bin - event_rate_bin)^2)
resolution = sum(weight_bin * (event_rate_bin - overall_event_rate)^2)
uncertainty = overall_event_rate * (1 - overall_event_rate)
brier_binned = reliability - resolution + uncertainty
```

This identity applies exactly to forecasts replaced by each bin's mean
probability. The raw probabilities generally have a different Brier score.
`brier_grouping_difference = brier - brier_binned` records that difference and
can have either sign. The report does not claim the raw Brier score equals the
three-term decomposition. Small floating-point differences are possible.

## Files and uncertainty

| File | Contents |
|---|---|
| `summary.csv` | Model/lead/group Brier scores, probability bias, event rates, decomposition, counts and bootstrap bounds |
| `bins.csv` | Every fixed probability bin, finite/missing counts, event rates, mean probabilities, scores and bounds |
| `daily.parquet` | Model/lead/group/day/bin input counts and sufficient statistics that reproduce each score |
| `manifest.json` | Source hash, declared event threshold, models, leads, binning, bootstrap settings and figure mapping |
| `reliability_NNN.png` | With `--plot`, pooled reliability and scored probability distributions for one model |

The bootstrap clusters all sites and bin counts within fixed calendar blocks
anchored at 1970-01-01. It resamples only blocks with scored rows, preserves
calendar gaps, and computes row-weighted rates from their sufficient totals.
There must be at least five scored blocks for bounds. Groups or bins with no
events, or events on every scored row, retain their scores but have missing
bounds and `bootstrap_status=constant_outcome`. Resampling cannot reveal event
variation absent from the sample; a zero-width interval is not evidence that
the underlying event frequency is exactly zero or one. This guard withholds
all three sets of bounds, including Brier and mean-probability bounds, rather
than presenting partial uncertainty for that degenerate outcome sample.

These are reporting guards, not proof of adequate data. Point groups can have many rows
but few blocks or smoke events. Bounds for event frequency, mean probability
and Brier score are marginal percentile intervals. They are not joint bounds
on the reliability curve, tests of calibration, or adjusted for multiple bins,
models and groups. Dependence between blocks and seasonal changes may remain.

The figure includes only pooled groups. Point, month and smoke results remain
in the tables. It places each nonempty bin at its scored mean probability,
retains asymmetric event-frequency bounds when available, and does not invent
intervals for sparse bins. The right panel shows the scored probability
distribution. Missing probabilities and targets remain in the tables.

Interval calibration and probability calibration are different properties.
CQR's outer-interval coverage does not establish reliable exceedance
probabilities, especially between or beyond the supplied quantiles. Use this
report with the [interval diagnostics](INTERVAL_DIAGNOSTICS.md),
[input-support audit](FEATURE_SUPPORT.md) and
[development calendar review](DEVELOPMENT_PLANS.md). Earlier use of the same
development dates still limits any claim about generalization.
