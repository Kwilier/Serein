# Audit a development NOAA cache archive

Use this when a feature rebuild needs many HRRR/AQM extractions checked at once.
The audit reads one extraction at a time, records its exact file hash, and uses
the same daily aggregation as the [single-file audit](INPUT_COVERAGE.md). It does
not download, repair, relabel, or overwrite a cache.

```powershell
uv run python -m serein.eval.cache_archive `
  --config configs/west_us.yaml `
  --cache-root data/interim/west_us `
  --points data/interim/west_us/points.parquet `
  --output development_cache_audit
```

The output directory must be new under an existing parent. The default enumerates
every configured season day in the training and calibration years, including
dates with no cached file. `--year 2023` limits the scan to that development year;
repeat it for more years. `--source hrrr` or `--source aqm` limits the sources.
Cycles, leads and issue hour come from the study configuration. HRRR files are
read from `hrrr/YYYYMMDD_HHz.parquet`; bias-corrected AQM files are read from
`aqm_bc/YYYYMMDD_HHz.parquet`, relative to `--cache-root`.

The command rejects reserved test years and overlapping training, calibration
and test years before opening an extraction. It only inspects inputs for the
selected issue days, including forecast steps stored inside those files. It
does not inspect target observations or execute a backtest. Keep using development
years until the team authorizes the reserved evaluation.

## Files and statuses

`files.csv` has one row per expected source/date, including failures:

| `file_status` | Meaning |
|---|---|
| `inspected` | Parquet decoded and its hourly inputs passed the aggregation checks |
| `missing` | No file at the expected path; every requested feature is recorded as missing rows |
| `unreadable` | A local I/O error prevented reading bytes; feature coverage is unknown |
| `invalid` | Parquet decoding, schema, date, forecast-step or aggregation validation failed; feature coverage is unknown |

The inventory retains the original error type and message for invalid or
unreadable files. A file hash and byte count are present whenever its bytes could
be read, even if decoding failed. Empty decoded caches are inspected and have
missing-row coverage, following the feature pipeline's behavior.

`registry_status` is independent of file validity. `match` means the extraction
declares the supplied registry's ordered IDs and coordinates. `mismatch` means
its stored hash differs. `unknown` means it has no stored point hash. Legacy
files remain unknown; finding their IDs in today's registry cannot establish
where they were originally sampled. `n_foreign_points` counts distinct cached
IDs outside the requested registry, including a missing ID when present.
Foreign rows do not become evidence of support for current registry points.
Timestamp and forecast-step checks apply to rows for requested IDs; foreign
rows are outside that validation scope.

`completion_status` preserves the extraction's declared completion flag:
`declared_complete`, `declared_incomplete`, `unknown`, or `invalid` for a
nonboolean flag. A declared complete extraction may still have unavailable
later local days because its forecast horizon is too short. None of these
metadata fields certifies the extraction's content independently.

`coverage.csv` groups every successfully inspected or absent source/date by
variable, lead, aggregation and availability status. `n_points` counts requested
points in that combination. It includes `missing_rows`, `no_finite_values` and
`insufficient_samples`, as well as `available`. Invalid or unreadable files have
no coverage rows because their feature coverage is unknown. Join against
`files.csv` before making an archive-wide percentage; silently omitting those
files would overstate availability. A completely invalid archive produces an
empty coverage table with its column headers intact.

`manifest.json` records the exact point-file and config-file hashes, registry
hash, dates, sources, cycles, leads, point count and file/provenance counts.
`status=complete` means the scan completed, including missing or invalid inputs.
A successful command exit does not mean every feature is available or every
cache can be reused. Compare `n_inspected_files` with `n_expected_files` and
inspect the status counts. Select a fresh output directory for a new audit.

## Limits and interrupted scans

Point coordinates and standard-time offsets come from the supplied registry.
The report counts input availability using the model's finite-sample thresholds,
not a requirement for contiguous hourly data. It does not validate units, public
posting times, raw GRIB provenance, observation completeness, smoke attribution,
model accuracy, or calibration. A registry hash that matches a stored declaration
does not prove independently that the coordinates were sampled correctly.

The tool streams small coverage summaries and keeps one decoded cache in memory.
An ordinary interruption or exception retains the inspected file records and
marks the manifest `interrupted`. A forced process termination can leave a
`running` manifest and incomplete or buffered tables. Such a directory is not a
completed audit and is never resumed automatically. Original caches remain
unchanged. Review unknown provenance and provider gaps before freezing results;
do not replace missing inputs with zeros or attach current hashes to old caches
as though their extraction provenance had been recovered.

## Development archive example, November 6, 2026

The local West US archive scan covered 459 issue dates in June through October
2021-23 and all 3,530 registered points, including grid locations. It inspected
913 files and recorded five missing bias-corrected AQM files:
2021-06-30, 2021-09-08, 2022-09-19, 2023-08-09 and 2023-09-18. No inspected file
failed schema or timestamp validation. All 913 present files lacked both a
registry declaration and a completion flag, so their provenance remains unknown.
No foreign point IDs were found.

The report retained 55,089,180 requested point/field/lead/aggregation combinations.
Every source/date/field/lead/aggregation count summed to the full registry size.
The fraction meeting the model's finite-sample requirement was:

| Input | Lead 0 | Lead 1 | Lead 2 | Lead 3 |
|---|---|---|---|---|
| HRRR smoke | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| AQM PM2.5 | 0.9804 | 0.9804 | 0.8768 | 0.0000 |

Each entry's denominator was 1,620,270 point/issue-day combinations, including
missing files. HRRR smoke at lead 2 had insufficient samples everywhere;
lead 3 had no rows. These are daily-input availability results, not errors from
a forecasting model. A model can still return finite long-lead forecasts by
using other inputs and imputation. Review these horizon limits when investigating
under-predicted peaks, before attributing the problem to an output distribution.
The legacy `noaa_best` baseline also falls back to persistence when both daily
NOAA fields are missing. A finite value in that column does not establish daily
NOAA input availability; see the [baseline definitions](../src/serein/models/baselines.py).

The installed-wheel command reproduced the August 2023 AQM inventory and coverage
exactly. Rehashing every inspected source file and the point registry after the
scan confirmed that their bytes were unchanged. Registry-file SHA-256:
`17b5a038639bdce6a2f87ff52ba3f541fc9f3811350c68b62201afe7839e191f`.
The audit does not establish why a file is absent, when a provider posted it, or
whether a current registry describes an old extraction correctly. It does not
replace the team's development or reserved-evaluation results.
