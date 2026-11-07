# Check development input support

Use this offline audit to compare inputs at a later development period or a new
set of points with a fixed reference table. For example, compare training inputs
with calibration inputs, then compare calibration inputs with scoring inputs.
These answer different questions. Familiar raw inputs do not establish forecast
accuracy or transfer of interval coverage.

```powershell
uv run python -m serein.eval.feature_support `
  --reference calibration_inputs.parquet --query scoring_inputs.parquet `
  --feature aqm_mean_L1 --feature hrrr_smoke_mean_L1 `
  --feature obs_prevday --feature hms_prev `
  --group point --group month `
  --out outputs/calibration_input_support
```

Choose features and the reference period before inspecting scoring outcomes.
The command requires an explicit list of input columns. It does not infer model
features, fit preprocessing, select a reference period, or use target columns.
If reviewing an ablation, include only the inputs that model actually uses.
Raw table features differ from the transformed inputs, missingness flags,
neighbor aggregates and history windows seen by the model.

Both files must contain unique `point_id`, `issue_date` rows. IDs must be
nonempty strings, and issue dates must be timezone-free calendar dates before
2024. Selected features must have real numeric dtypes, with finite values or
missing entries. Infinities, complex values and overflow are rejected. Negative
inputs are retained without clipping. The tool does not check physical units or
whether a selected column was actually available at forecast issue time.
Target columns beginning with `y_L` and identifier columns cannot be selected.
Other unselected columns are ignored, including observations and forecasts.

Only the two named files are opened. A `FINAL_RUN_LOCK.json` alongside either
input is refused. The output directory must be new. This command does not open
the reserved evaluation, train models, change calibration, or modify inputs.
All rows are included regardless of `kind`; filter the inputs explicitly to
compare EPA monitors with EPA monitors or grid points with a chosen reference.

## Reference population

The default `--reference-scope pooled` compares each query group with the same
full reference population. A query monitor can look unusual relative to the
pooled population even when its conditions are typical for that monitor.

`--reference-scope point` instead uses only reference rows for each query point.
This can expose changes hidden by a broad pooled range. A point absent from the
reference has `no_reference_values`; it is retained, with zero reference rows,
rather than compared against a substituted population. Reference-only points
do not get individual rows in this mode. The manifest still counts all input
rows, including reference-only points. Point groups in this mode repeat the
corresponding point's `all` comparison. Month groups compare that query month at
each point with that point's full reference period, not a matching month.

`seen_reference_point` only means an ID occurred in the reference. It does not
mean that any particular feature or calibration target was available there.
An entirely missing reference feature is reported separately from an unseen ID.

## Files and quantities

| File | Contents |
|---|---|
| `reference.csv` | Reference counts, finite-day counts, mean, population standard deviation and empirical quantiles in raw units |
| `shift.csv` | Per-feature distribution comparisons for all rows and requested point/month groups |
| `rows.parquet` | Query keys, seen-point flag and counts of selected features that are missing, unsupported or outside reference ranges |
| `summary.csv` | Query rows and days, unfamiliar-point counts and counts of rows with any feature flag |
| `manifest.json` | Exact input-byte hashes, selected settings, overlapping point/issue keys and interpretation limits |

Each finite reference or query row has equal weight. A site with more available
days contributes more than one with fewer days. `n_reference_finite_days` and
`n_query_finite_days` count distinct issue days with finite values, not
independent observations. Missing values contribute to missingness rates but
are excluded from quantiles, moments and distribution distances.

`ks_distance` is the largest absolute difference between the two empirical
cumulative distribution functions, including tied values. It ranges from zero
to one. It is a descriptive distance; the report does not apply a significance
test that treats correlated monitor-days as independent samples.

`wasserstein` is the one-dimensional Wasserstein distance in the feature's raw
units. `wasserstein_ref_sd` divides that distance by the reference population
standard deviation. `mean_shift_ref_sd` is the signed query mean minus reference
mean, divided by the same reference standard deviation. These normalized
quantities are missing for constant reference features, while the raw distance
and range flags remain available. No small standard deviation floor is added.

`missing_rate_change` is query missing fraction minus reference missing fraction.
The below-minimum and above-maximum rates use finite query values as their
denominator. `outside_reference_90_rate` counts finite query values strictly
outside the reference's empirical 5th/95th percentiles. Quantiles use NumPy's
default linear interpolation. Values equal to a bound are inside. This is not
a nominal 10% anomaly test, especially with tied or discrete inputs.

Row flags count features, not their magnitudes. A missing query value is not
also counted as out of range. A reference feature with no finite values adds
to `n_features_without_reference` for every relevant query row. Its range flags
remain zero because no range can be estimated. Read missingness and support
columns together; zero range flags alone do not mean adequate support.

## Review alongside forecast performance

Use the [interval report](INTERVAL_DIAGNOSTICS.md) to inspect coverage at the
same monitors and months, and the [paired comparisons](FORECAST_COMPARISON.md)
to inspect accuracy. Keep their populations and keys explicit. Large input
shifts can motivate investigation; they do not explain a coverage failure by
themselves. Similar marginal distributions can conceal unfamiliar combinations
of inputs, graph structure, sensor placement or event histories.

The audit reports overlapping reference/query keys without treating overlap as
an error. A self-comparison is useful for checking the export, but its small
distances provide no evidence of generalization. The manifest does not verify
that the reference actually trained or calibrated a particular model.
The [development planner](DEVELOPMENT_PLANS.md) checks declared role calendars
and observed target overlap separately. Neither tool restores an untouched
holdout when development dates have already been used for tuning.
