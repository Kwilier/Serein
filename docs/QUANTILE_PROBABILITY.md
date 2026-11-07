# Audit probability assumptions between forecast quantiles

SmokeCast exports five concentration quantiles. Its warning conversion
interpolates a CDF between them and assigns half the remaining tail mass
outside them. Those choices supply information that the five quantiles alone
do not determine. This offline report brackets an event probability using the
quantile inequalities and checks fixed warning cutoffs across that bracket.
It does not change inference, calibration, probabilities or warning cutoffs.

```powershell
uv run python -m serein.eval.quantile_sensitivity `
  --predictions development_quantiles.parquet `
  --model smokecast --lead 1 --lead 2 --lead 3 `
  --cutoff 0.1 --cutoff 0.25 --cutoff 0.5 --cutoff 0.75 --cutoff 0.9 `
  --group point --group month --group smoke `
  --out outputs/development_quantile_probability_review
```

Use a new directory. The source hash records the bytes actually decoded; the
source remains unchanged. Selected issue and target dates must precede 2024,
and a `FINAL_RUN_LOCK.json` next to the input is refused. The command does not
run the reserved evaluation.

## What the bracket means

Assume nonnegative concentrations and valid quantiles in the usual generalized
inverse sense: `q(a) = inf{x: F(x) >= a}`. For the inclusive event `Y >= t`:

* If `q(a) < t`, then `P(Y >= t) <= 1 - a`.
* If `q(a) >= t`, then `P(Y >= t) >= 1 - a`.

The report takes the largest available lower bound and the smallest available
upper bound. Without an applicable quantile, the bound is zero or one. These
are conservative brackets and can include unattainable endpoints. At `t = 0`,
known nonnegative concentration support gives `[1, 1]` for complete rows.

The inequalities also follow from the population-quantile coverage conditions
in the [NumPy quantile reference](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#notes),
checked on 2026-11-06: `P(Y < q(a)) <= a` and `P(Y <= q(a)) >= a`.

For example, a threshold strictly between the median and 75th percentile has
bracket `[0.25, 0.5]`. A threshold above the 95th percentile has bracket
`[0, 0.05]`; the shipped conversion's value of 0.025 is one choice inside it.
Equality follows the inclusive event: if several quantiles equal the
threshold, the smallest level at that value supplies the strongest lower
bound. A continuous-CDF or rightmost-tie conversion can disagree with that
convention at an atom. The report records exact equality and conversion
disagreements for review rather than changing the conversion.

These are **conditional mathematical implications**, not uncertainty intervals
for the real probability. Learned quantiles can be inaccurate. CQR coverage
of the outer interval does not establish that every level is a valid
conditional quantile. Clipping or collapsing quantiles can also affect their
interpretation. A decision labeled `forced` below is stable only under these
assumptions. It is not a guarantee about a child's risk or an actual event.

## Export all levels in concentration units

The input shares the [probability report](WARNING_RELIABILITY.md) keys:
unique nonempty string `point_id`, timezone-free midnight `issue_date`, and
numeric observed concentrations `y_L<lead>`. Include every selected model and
lead with columns `<model>_q<index>_L<lead>`. Indices start at zero and refer to
the supplied ordered level list, not percentage values. By default:

| Index | Level | Example column at lead 1 |
|---|---|---|
| 0 | 0.05 | `smokecast_q0_L1` |
| 1 | 0.25 | `smokecast_q1_L1` |
| 2 | 0.50 | `smokecast_q2_L1` |
| 3 | 0.75 | `smokecast_q3_L1` |
| 4 | 0.95 | `smokecast_q4_L1` |

Use repeated `--level` flags to replace this list for another quantile grid.
Levels must increase strictly and lie in `(0, 1)`. Quantiles must be
nonnegative physical concentrations in micrograms/m3, not log1p values. For
example, given calibrated `q_log` with shape `[rows, leads, levels]`:

```python
import numpy as np
from serein.eval.backtest import prob_exceed

export = rows[["point_id", "issue_date", *[f"y_L{x}" for x in leads]]].copy()
for index, lead in enumerate(leads):
    for qi, level in enumerate(levels):
        export[f"smokecast_q{qi}_L{lead}"] = np.expm1(q_log[:, index, qi])
    export[f"smokecast_prob_L{lead}"] = prob_exceed(q_log[:, index], levels, thr=35.5)
export.to_parquet("development_quantiles.parquet", index=False)
```

Retain model, preprocessing, calibration, input and level provenance. A median
and two outer bounds cannot reconstruct all five levels. The optional
`<model>_prob_L<lead>` columns compare an existing conversion with the bracket;
they are not required and are never inferred. Missing observations are allowed
because bracket sensitivity does not depend on realized outcomes. Realized
event counts remain available for groups with observations.

Any missing quantile makes the row incomplete and leaves both bracket limits
missing. Known values are still checked for crossings across missing entries.
Infinities, negative concentrations, complex/nonnumeric data and crossed known
quantiles fail. Supplied probabilities must be finite in `[0, 1]` or missing.
`--group smoke` additionally requires `hms_prev` with 0, 1, 2, 3 or missing;
missing labels form an `unknown` group.

## Read the decision audit

For each fixed probability cutoff `c`, warnings use `p >= c`:

| Summary | Condition on complete rows |
|---|---|
| `n_forced_warning` | Bracket lower bound is at least `c` |
| `n_forced_quiet` | Bracket upper bound is strictly below `c` |
| `n_ambiguous` | Neither condition holds |

These counts partition `n_complete`. An upper bound equal to `c` stays
ambiguous unless the lower bound also reaches `c`. `ambiguous_fraction` uses
complete rows, including rows without observations. This isolates conversion
assumptions from outcome availability. Point/month/smoke groups describe the
exported rows only, with no bootstrap or probability recalibration.

`n_probability` counts complete rows with a finite supplied probability.
`n_outside_bracket` counts those probabilities outside the bracket, regardless
of their warning decision. `n_warning_disagreement` counts a supplied warning
opposite to a forced decision at that cutoff. An outside probability can still
produce the same warning at some cutoffs. Missing probabilities stay in the
row audit but are excluded from these comparisons. All comparisons use the
exact supplied numeric values; inspect numerical precision near boundaries.

`n_observed` and `n_events` count complete rows with finite observations and
observed events. The ambiguous/forced event counts use that same target
availability. A forced-quiet observed event is a forecast miss, not a
contradiction of the probability bracket's mathematical definition.

| File | Contents |
|---|---|
| `summary.csv` | Per-model/lead/group/cutoff support, stable/ambiguous decisions, tail/equality counts and conversion comparisons |
| `rows.parquet` | Keyed physical-event observations, bracket limits, completeness, optional probability and conversion flags |
| `manifest.json` | Decoded source hash, concentration units, level-index mapping, cutoffs, groups and assumptions |

Review this report with [probability reliability](WARNING_RELIABILITY.md) and
[event discrimination](EVENT_DISCRIMINATION.md). Bracket width diagnoses what
the quantile grid leaves unspecified under its assumptions. It does not
measure forecast accuracy, validate probability calibration, or restore an
untouched development holdout.
