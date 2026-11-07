# Review a separate development schedule

The [split audit](SPLIT_AUDIT.md) checks the current alternating-week development
schedule. Its illustrative purge can leave too few validation days. Use this
planner to inspect an explicit alternative with four separate roles before
changing any fitting code:

| Role | Intended use after team review |
|---|---|
| `train` | Fit model parameters |
| `tune` | Select epochs and development hyperparameters |
| `calibrate` | Fit interval corrections and warning cutoffs |
| `score` | Measure development performance |

This tool does not adopt a schedule, fit or score a model, or change the backtest.
In particular, the current backtest still combines tuning and calibration in its
validation role. Exporting this plan does not make that behavior change.

## Supply dates, not model results

Save a local JSON file with exactly the four roles above. Each role contains one
or more inclusive issue-date ranges. For example, this is a **hypothetical plan**,
not a recommendation or the team's protocol:

```json
{
  "train": [
    {"start": "2021-06-01", "end": "2021-10-31"},
    {"start": "2022-06-01", "end": "2022-10-31"}
  ],
  "tune": [{"start": "2023-06-01", "end": "2023-07-10"}],
  "calibrate": [{"start": "2023-07-15", "end": "2023-08-20"}],
  "score": [{"start": "2023-08-25", "end": "2023-10-31"}]
}
```

Run against the saved training and development feature tables:

```powershell
uv run python -m serein.eval.split_plan `
  --train-table data/processed/west_us/table_train.parquet `
  --calibrate-table data/processed/west_us/table_calibrate.parquet `
  --plan development_plan.json `
  --gap-days 1 `
  --plot `
  --out outputs/development_plan_review
```

The output directory must be new. The command reads only the three named input
files, records hashes of the decoded bytes, and preserves the originals. It does
not discover or open the reserved test table. Dates in either input table must
precede 2024, including requested target dates, even for unassigned or non-EPA
rows. Declared target dates must also precede 2024 even when no input rows exist
for them. A lead-3 issue on December 31, 2023 is therefore refused. This stricter
guard avoids a proposed development plan requesting reserved target dates.

Only EPA rows are assigned when `kind` exists. Custom tables without `kind` use
all rows. The two input tables are pooled, so ranges can select data from either
source; `source_table` in the assignments identifies the original table. Duplicate
point/issue-date rows across sources are rejected instead of silently choosing
one copy. Point IDs must be nonempty strings. No implicit ID conversion occurs.
CSV readers can infer numeric IDs and drop leading zeros. With pandas, use
`dtype={"point_id": str}` when reading report files containing monitor IDs.
Input columns named `role` or `source_table` are reserved and rejected.

## Inspect counts and conflicts together

| File | Review question |
|---|---|
| `report.json` | Are roles nonempty, are all role/lead targets present, and are calendars separated? |
| `plan.json` | Which explicit ranges were checked? |
| `ranges.csv` | How many declared issue days have no EPA rows at all? |
| `roles.csv` | How many rows, days, finite targets and unique monitor-day labels remain? |
| `availability.csv` | How many finite and missing targets does each role/lead have? |
| `point_availability.csv` | Which monitors have few or no targets for a role/lead? |
| `point_support.csv` | Which role/lead pairs have different monitors with finite targets? |
| `role_order.csv` | Do all declared targets follow the train, tune, calibrate, score order? |
| `calendar_conflicts.csv` | Which role pairs violate the requested target-date gap? |
| `overlaps.csv`, `overlap_days.csv` | Which finite monitor-day labels are actually reused? |
| `assignments.csv` | Which source, monitor and issue date was assigned to each role? |

With `--plot`, `development_plan.png` shows pale bands for declared issue-date
ranges and colored marks for days with EPA rows. Years inside a declared range
are shown even if all their rows are missing. The lower panel shows the fraction
of available rows with finite targets for each role/lead. It excludes completely
absent rows from the denominator; use `ranges.csv` and the monitor tables to
inspect those gaps. The figure is descriptive and does not declare a plan ready
for fitting.
PNG export uses a file-only canvas and does not require a desktop GUI backend.

Assignments include only available rows inside the declared ranges. Rows outside
every range are counted as unassigned, rather than silently added to training.
Empty roles remain in summaries and have zero finite targets. A monitor entirely
absent from a role has no row in that role's point table; compare monitor IDs
across roles as well as reading individual counts. An absent issue day means no
EPA rows exist for that day; it does not mean every monitor is complete on other
days. Different ranges of one role can cover separate seasons.
The point-support table compares monitors with at least one finite target at
each lead. It reports intersections and monitors present only in one role,
including score monitors with no finite calibration target. It does not certify
that any monitor has enough calibration data.

The calendar check expands every declared issue day and requested lead, including
days absent from the feature table. For each pair it counts issue days whose
targets are within `gap_days` of any target in the other role. A gap of zero
forbids identical target dates; a gap of one also forbids adjacent target dates.
Gaps use calendar days, not row positions. Sparse leads such as `--leads 1 3`
use only those two horizons. The tool never removes conflicting days for you.

Finite-label overlap follows the [split audit's definitions](SPLIT_AUDIT.md).
Missing observations can remove observed shared labels while leaving a calendar
conflict. Different monitor populations can do the same. Small disagreements
between repeated observations use that audit's numeric tolerance; conflicting
observations are rejected. Negative observations are counted without clipping.

## What this can establish

A plan can have nonempty roles and separated targets while still having too few
independent smoke events or poorly matched seasons, geography and exposure
levels. Counts are descriptive, not a statistical adequacy test. A range is not
automatically restricted to configured fire seasons, and the planner does not
require forward-in-time ordering between roles. Feature availability, spatial
dependence, prior model tuning and checkpoint provenance are outside this check.
`forward_ordered` checks the strict train, tune, calibrate, score target-date
order using every declared range. It can be false when calendars are separated,
for example when an early scoring range precedes tuning. It does not establish
that an observation was published before a later forecast was issued.

Choose schedules before looking at their model scores. Record the reason for
each boundary and any event, seasonal or monitor-population tradeoffs.
Reassigning development dates does not undo earlier tuning on those dates or
turn them into an untouched evaluation set.
Inspect the source hashes and counts with the team before changing the protocol. Adopting
a plan requires implementing separate fitting/calibration/scoring inputs and
rerunning development. This report cannot authorize the reserved evaluation or
establish improved forecasting accuracy.

If a write fails, a partial new output directory may remain. Preserve it for
diagnosis and use a new destination on retry. The command does not overwrite an
existing report.

## Local verification

```powershell
uv run pytest tests/test_split_plan.py tests/test_split_plan_plots.py tests/test_split_audit.py tests/test_split_plots.py -q
```

The planner tests cover complete and empty roles, cross-lead reuse, missing days,
missing targets, sparse horizons, multiple seasons, duplicate source rows,
reserved dates, source preservation and a separate calendar-set oracle. Software
checks establish report behavior, not the adequacy of a proposed schedule.
