# Development target-label overlap

Run this check before interpreting development scores or freezing the analysis:

```powershell
uv run python -m serein.eval.split_audit `
  --train-table data/processed/west_us/table_train.parquet `
  --calibrate-table data/processed/west_us/table_calibrate.parquet `
  --out outputs/split_audit_dev
```

The output directory must be new. The command reads only the two named Parquet
files. It never trains, scores, downloads data, changes the input tables, or reads
the reserved test table. Issue years 2024 and later are refused. Source paths and
SHA-256 hashes identify the exact bytes decoded. For custom target columns, pass
the matching list, for example `--leads 1 2 3`.
Add `--plot` to save `split_audit.png`, a local calendar figure. Colored bars
show retained issue days, gray crosses show proposed exclusions, and the lower
panels compare shared-label counts on target dates. Every issue or shared-target
year gets a panel, including gaps between seasons. The figure does not refit a
model or change the assignments. A write failure can leave a partial new output
directory; keep it for diagnosis and retry with a new destination.
When a `kind` column is present, only `epa` rows enter the development roles,
matching the backtest. The report counts excluded rows. Custom tables without
`kind` use all rows. Identifier and issue-year checks still cover the full inputs.

## Why issue-week separation is not target separation

Development mode splits the calibration table by ISO issue week: even weeks
provide early stopping, conformal calibration and warning-cutoff tuning; odd
weeks provide scores. An issue date and its target date differ by the lead.
For example, a Sunday lead-1 forecast and a Monday lead-0 forecast use the same
Monday observation at the same monitor. Putting Sunday and Monday in different
roles does not make those labels different.

The current network's early-stopping loss uses all leads. CQR corrections and
warning cutoffs are fitted per lead, so cross-lead overlap alone does not show
that a lead's own cutoff used its scored labels. Inspect the role of each reused
label before drawing conclusions about any metric.

The audit matches finite target labels by `(point_id, issue_date + lead)`, across
all requested leads. A label is counted once per role, even when several issue
dates or leads reuse it. An exposure counts each issue/lead use of that label.
Missing targets remain in the role totals but cannot establish shared labels.
Negative observations are retained and counted separately, without changing
the training code's clipping rule. Infinite values, duplicate monitor/issue-date
rows, invalid dates and conflicting observations for one monitor/target date
are rejected. Small floating-point differences are allowed with `rtol=1e-6`
and `atol=1e-6`.

This is a label-reuse audit. It does not estimate how much reuse affects a score
or prove that any result is invalid. It does not audit feature availability,
sensor exclusion, checkpoint provenance, previous tuning, or spatial dependence.
It does not check configured split years or season membership beyond the reserved
issue-year guard. Inspect the source hashes and dated assignments as well.
An earlier target legitimately used as a later forecast's available observation
is a different question from sharing labels across fitting and scoring roles.

## Files and counts

| File | Contents |
|---|---|
| `roles.csv` | Rows, issue days, finite/missing/negative target slots and unique labels per role |
| `overlaps.csv` | Shared labels, target days and exposures for each role pair |
| `overlap_days.csv` | Shared-label counts by local target date and role pair |
| `issue_assignments.csv` | Original roles and a proposed keep/exclude flag for each issue day |
| `proposed_*.csv` | The same three audit tables after the proposed validation purge |
| `report.json` | Inputs, hashes, settings, removal counts and limitations |

All monitor IDs must be nonempty strings, preserving leading zeros. Dates must
be timezone-free calendar dates at midnight. Input indexes do not affect results.
Labels at different monitors are distinct, even on the same target day. The sum
of pairwise overlap counts can count a label repeatedly if three roles share it.
There is no forecast-error metric in this report.

## An illustrative purge, not a new default

The proposal keeps the training table and odd-week score schedule fixed. It
removes an entire even-week validation issue day whenever any of its requested
target dates touches a score target date. This calendar rule applies to every
monitor, including missing observations. A missing label today should not
silently become permission to reuse it when a cache is rebuilt.

Add `--gap-days 2` to require candidate target dates to be more than two calendar
days from any score target. Gaps are measured in calendar days, not row positions.
The purge supports sparse or nonconsecutive lead lists and does not assume that
all intermediate leads exist.

The proposed assignments are a review aid. The command does not apply them to a
backtest. Alternating weeks with several leads can leave very little validation
data; `proposal_has_validation_rows` says only whether any rows survive, not
whether there are enough for useful fitting or calibration. Training-role
overlaps are reported but not repaired. Inspect `proposed_overlaps.csv` rather
than assuming the proposal removes every overlap.

Early stopping, conformal fitting and warning tuning still share the combined
`validation` role. The purge does not separate those uses. Nor does zero shared
labels establish independence between neighboring pollution days. The team must
choose an adequate schedule, calibration population and any additional temporal
gap before changing the protocol, then rerun development. This check does not
freeze the protocol or authorize the reserved evaluation.
It also does not turn retrospective alternating-week development into a
forward-in-time evaluation.

On the rebuilt local development tables checked November 6, 2026, leads 0 through
3 shared 20,752 monitor-day labels between validation and scoring, across 64
target dates. The nominal calendar purge retained 11 of 76 validation issue
days (4,675 of 32,300 rows). Adding a one-day gap retained no validation days.
These counts were independently checked with Python label sets. The training
and calibration file SHA-256 prefixes were `fc347c674c7e` and `2d03fda7fe34`;
the command writes their full hashes. These are input-audit results, not an
estimate of score bias or a replacement for the historical forecast results.
Counts depend on the requested leads and exact table bytes. Auditing only a
subset of leads does not establish separation for a model fitted or selected
using additional leads.

In a separate 20-epoch, five-seed development probe on these tables, the original
and purged schedules selected exactly the same model weights and produced
identical median forecasts. Interval calibration and warning cutoffs changed
with the smaller fitting population. This illustrates why label counts alone
cannot establish score bias; it is not a full-budget rerun.

## Python use and verification

`audit_splits({"train": train, "validation": val, "score": score}, leads=[0, 1, 2, 3])`
returns role totals, pairwise overlaps and daily overlaps. Empty roles are allowed
and remain visible. `purge_issue_days(candidate_dates, reference_dates, leads=...,
gap_days=...)` returns a positional boolean mask without mutating either schedule.
The generic functions can audit other research schedules; the command's
development-only year guard belongs to `development_roles`.

```powershell
uv run pytest tests/test_split_audit.py tests/test_split_plots.py -q
```

The tests include cross-lead reuse, distinct monitor identities, missing and
negative targets, conflicting labels, sparse horizons, calendar gaps, empty
roles, reserved-year rejection, source preservation and a separate calendar
oracle for the proposed purge.
