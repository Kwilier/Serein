# Optional fire-history features

Task B-16 proposes fire-growth or transport inputs beyond the existing 24-hour
FRP sums. `serein.features.fire_history` builds an optional development input:
the change in nearby detections and fire radiative power between two adjacent
acquisition windows. It does not estimate burned-area growth, plume age, smoke
arrival time, or emissions. No forecast improvement has been established and
B-16 remains open.

The normal table builder and saved models are unchanged. This module can produce
a separate point summary or append numeric columns to a copy of a development
table for an explicit experiment.

## Required inputs

All inputs are local Parquet files. Naive timestamps mean UTC; timestamps with
offsets are converted to UTC.

| Input | Required columns | Meaning |
| --- | --- | --- |
| Detections | `source`, `latitude`, `longitude`, `time_utc`, `available_utc`, `frp` | Acquisition time, first availability time, and FRP in MW |
| Coverage | `source`, `start_utc`, `end_utc`, `available_utc`, `lat_min`, `lat_max`, `lon_min`, `lon_max` | A declaration that the source archive for `(start_utc, end_utc]` inside these bounds was fully retrieved by `available_utc` |
| Points | `point_id`, `lat`, `lon` | Unique point IDs and coordinates in degrees |
| Development table | `issue_date`, `point_id`, `lat`, `lon`, plus its existing columns | Calendar issue dates with unique point/date pairs |

Each run selects exactly one source. Use distinct source identifiers for products
and instruments with different retrieval or sampling behavior. Do not pool FIRMS
standard processing with near-real-time products, or VIIRS with MODIS or GOES.
Repeated GOES observations would measure a different sampling process.

The current FIRMS and HMS caches do not establish historical `available_utc` or
complete source coverage. The module deliberately does not infer them. Acquisition
time, present-day download time, and file modification time do not prove that a
record was available for a historical forecast. Existing caches therefore need
additional provenance before a historical availability claim is justified.

A coverage declaration describes archive retrieval, not cloud-free satellite
observations, overpass completeness, or absence of fire. Bounds must describe the
actual retrieved region, not the locations of the detections it happens to
contain. A source with no declaration is missing, even when its file is empty.

Each point's entire great-circle search neighborhood must fit inside a declared
rectangle for that rectangle's interval to count. This includes fires outside
the study boundary. Time intervals from different rectangles can combine if
each contains the entire neighborhood. Spatial tiles are not assembled into a
larger footprint; this conservative rule can leave a tiled archive incomplete.
Rectangles must have increasing bounds in degrees. Neighborhoods crossing the
date line or reaching a pole require full longitude coverage from -180 to 180.

## Windows and outputs

For issue time `T`, window length `W` and acquisition lag `A`, let `E = T - A`.
The recent window is `(E - W, E]` and the previous window is
`(E - 2*W, E - W]`. Default `W` is 24 hours, `A` is zero hours, and the
radius is 250 km. Choose a lag consistent with the archive's demonstrated
availability; the module does not choose one from a product name.

Only detections and coverage declarations with `available_utc <= T` are used.
Detections acquired after `E` are excluded, even if already available. Coverage
intervals can overlap or meet at their boundaries. Any gap in time or inadequate
spatial coverage leaves that point's window incomplete. Coverage cannot be declared available before its end, and a detection
cannot be available before acquisition.

| Column | Meaning |
| --- | --- |
| `fire_recent_complete`, `fire_previous_complete` | Whether available declarations cover the entire acquisition window |
| `fire_recent_count`, `fire_previous_count` | Number of unique nearby detections in each complete window |
| `fire_recent_frp`, `fire_previous_frp` | Sum of nearby detection FRP in MW in each complete window |
| `fire_latest_age_hours` | Hours from issue time to the latest nearby detection in the complete recent window |
| `fire_frp_log_change` | `log1p(recent FRP) - log1p(previous FRP)` |
| `fire_count_change` | Recent count minus previous count |

Incomplete windows yield missing counts and FRP, not zeros. Complete empty
windows yield zero counts and FRP, with missing detection age. A missing nearby
FRP value makes the window's FRP sum missing while retaining the count and age.
Negative or infinite FRP and invalid active coordinates are rejected. Exact
duplicates from overlapping downloads count once. Conflicting FRP values for the
same source/location/acquisition are rejected after the availability cutoff.
Nearby means great-circle distance less than or equal to the configured radius.

For example, recent FRP 30 MW and previous FRP 5 MW produce
`log1p(30) - log1p(5)`, a positive change. Reversing the windows makes the change
negative. These are sums of sampled detections, not time-integrated energy.
Overpass timing, cloud cover, changing retrieval coverage, and instrument
sensitivity can change this proxy without a fire growing or shrinking.

## Run on a separate output

For one issue and a point file:

```powershell
uv run python -m serein.features.fire_history `
  --fires path/to/detections.parquet --coverage path/to/coverage.parquet `
  --points path/to/points.parquet --source VIIRS_SNPP_NRT `
  --issue-time 2023-07-03T18:30:00Z --acquisition-lag-hours 3 `
  --out outputs/fire_history/points.parquet
```

To append columns to a development table:

```powershell
uv run python -m serein.features.fire_history `
  --fires path/to/detections.parquet --coverage path/to/coverage.parquet `
  --table path/to/development_table.parquet --source VIIRS_SNPP_NRT `
  --issue-hour-utc 18.5 --acquisition-lag-hours 3 `
  --out outputs/fire_history/development_table.parquet
```

Use the study's actual UTC issue hour. The table mode preserves original columns,
row order, index and targets, including the existing `fire_dist_km` feature. It
rejects collisions with its history feature names to avoid overwriting an earlier
experiment. Single-issue output also includes source,
issue time, acquisition cutoff, radius and window settings. Both CLI modes write
an adjacent `.parquet.json` manifest with input/output hashes and parameters,
and refuse an existing output or manifest.

The Python APIs are `fire_history_features(...)` and
`augment_fire_history(..., issue_hour_utc=...)`. API callers must retain their
own source, parameter and provenance record.

## Before measuring benefit

1. Establish historical source availability and archive coverage. Do not fill
   missing dates with zero detections or declare coverage from file presence alone.
2. Append the same fixed-source feature definitions to copies of the training,
   tuning, calibration and scoring tables, using a reviewed development schedule.
   Do not alter the reserved evaluation inputs.
3. Fit preprocessing only on training data. Keep `fire_frp_log_change` and
   `fire_count_change` signed. Their `fire_` prefix avoids the existing `frp_`
   transform that clips negative inputs before `log1p`.
4. Compare matched seeds and populations, including peak errors at leads 2 and 3,
   warning tradeoffs, missingness, and per-source sampling differences. For an
   ablation that removes every fire input, exclude `fire_` as well as the
   existing `frp_` and `fire_dist` prefixes. The default development runner's
   fire ablation predates these optional columns and does not remove them.
5. Supply the same history features for every forecast input before using an
   experimental artifact outside its development run. The default daily and map
   table builders do not generate these columns. Existing preprocessing imputes
   missing columns, so an inference call succeeding does not establish that the
   new features were supplied. Serving integration and map-location coverage
   remain separate work.

Software tests verify the feature calculations and availability behavior. They
do not establish transport physics, historical provenance, or predictive value.

## Source checks

NASA's [active-fire attribute documentation](https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs)
defines acquisition time as the satellite overpass time in UTC and FRP as
pixel-integrated radiative power in MW. Its
[FIRMS FAQ](https://www.earthdata.nasa.gov/data/tools/firms/faq), read November 7,
2026, distinguishes rapid-delivery products from standard monthly-location
archives, which it says usually arrive with a two-to-three-month lag. An NRT
latency description therefore does not establish availability of a standard
archive at a historical issue time.

The FAQ also discusses overpass timing, missing observations and standard
products with additional quality information. Exact-row deduplication here only
removes repeated downloads of the same detection. It does not correct overlapping
orbit sampling or make a detection-count trend a physical fire-growth estimate.
