# Inspect cached NOAA feature availability

An extraction can contain every point ID and still lack a smoke field, a weather
variable, or enough samples for a local day. The model can return a finite forecast
after filling those missing inputs. Check availability separately from forecast
accuracy.

The offline audit reads one local HRRR or AQM extraction and its point registry.
It does not download data, load model weights, score observations, or change the
source files. For example, from the repository root:

```powershell
uv run python -m serein.eval.input_coverage `
  --source hrrr --cache data/interim/west_us/hrrr/20230818_12z.parquet `
  --points data/interim/west_us/points.parquet `
  --date 2023-08-18 --cycle 12 --leads 0 1 2 3 `
  --output hrrr_input_audit
```

For AQM, use `--source aqm` and the corresponding file under `aqm_bc/`. Supply the
actual extraction date and cycle. Extractions stored by the downloaders do not
contain an issue-date column, so the audit applies the supplied date and then
checks it against every valid timestamp and forecast step. It refuses inconsistent
dates, times, steps and nonempty files missing the hourly schema.

The output directory must be new, under an existing parent. The output contains
`input_coverage.parquet` and `summary.json`. A failure while writing can leave a
partial new directory; use a new destination on retry. The command's successful
exit means the audit was written. Inspect its statuses and provenance before
using the cache.

## Read the report

The report has one row for every requested point, variable and lead, including
missing combinations. An empty cache is reported as missing data. A point's fixed
`gmt_offset` defines its local standard day, using the same aggregation function
as the feature pipeline. Timestamps are interpreted in UTC before applying that
offset.

| Column | Meaning |
|---|---|
| `issue_date`, `valid_date`, `lead` | Input model-run date, local target date, and their difference in days |
| `source`, `variable` | `hrrr` smoke/weather or `aqm` PM2.5 |
| `aggregation` | `local_day` or the separate `issue_wind` input |
| `n_timestamps` | Distinct supplied UTC timestamps, including nonfinite samples |
| `n_finite` | Distinct timestamps with a finite value after duplicate averaging |
| `min_count` | Pipeline sample threshold: 12 for smoke/AQM, 4 for weather, 1 for issue-time wind |
| `mean`, `max` | Aggregated values when the threshold is met, otherwise missing |
| `status` | `available`, `missing_rows`, `no_finite_values`, or `insufficient_samples` |

Sample counts are not necessarily hours of coverage. In the default extraction,
smoke and AQM are hourly, while HRRR weather is sampled every three hours. Duplicate
rows at the same forecast timestamp are averaged before counting and daily
aggregation. NaN and infinite values do not meet the finite-sample requirement.
The audit does not enforce a contiguous sampling schedule beyond those pipeline
thresholds.

Issue-time wind uses the forecast hour selected by the feature pipeline,
`int(issue_hour_utc - cycle)`, for `u10` and `v10`. Its `lead=0` is a label for
that input, not a daily average. Use `--issue-hour-utc` if the study differs from
the default 18.5 UTC. Values retain the extraction's units, including ug/m3 for
PM2.5, m/s for wind and Kelvin for temperature; the audit does not convert units.

Short forecast horizons can leave later local days below the threshold even
with a complete extraction. These gaps remain visible. They are not evidence
that a download failed, nor should missing values be replaced by zero.

### Verified development example, November 6, 2026

Fresh 12Z extractions for August 18, 2023 were sampled at 15 Sacramento-area
two-kilometer cells. The export produced 60 grid forecasts and 510 input-audit
rows. The audit found the same pattern at each cell:

| Input | Day 0 | Day 1 | Day 2 | Day 3 |
|---|---|---|---|---|
| HRRR smoke and daily weather | available | available | insufficient samples | missing rows |
| AQM PM2.5 | available | available | available | insufficient samples |

Both issue-time wind components were available. The model still returned finite,
ordered quantiles at every lead, so forecast finiteness alone would have hidden
the later-day gaps. This used a five-member, two-epoch development fit to check
the export path. Forecast skill and interval calibration at these locations
remain unverified.

## Check provenance and limits

The JSON summary records the exact source-file SHA-256 hashes, requested
parameters, point counts by status, and the extraction's completion flag when
present. `extraction_registry_match` compares the extraction's stored point hash
with the supplied registry, including point order and coordinates:

- `true`: the extraction declares the same registry.
- `false`: it declares a different registry. Do not use the report as evidence
  that the current locations have been extracted.
- `null`: the extraction did not supply a point hash, so its provenance is unknown.

The standalone audit preserves stale or absent provenance in the report so it can
diagnose an old cache. The [neighborhood forecast export](NEIGHBORHOOD_MAPS.md)
requires matching hashes and all grid IDs before inference. It also writes this
availability report for every required history day, with `history_back=0` for
the current issue date.

This report audits NOAA-derived inputs, including variables a particular model
may have dropped. It does not establish observation, HMS or fire-data availability,
whether a run was publicly posted by the study's issue time, imputation effects,
spatial forecast skill, or calibrated uncertainty at new locations. Keep these
checks separate from the reserved evaluation.
