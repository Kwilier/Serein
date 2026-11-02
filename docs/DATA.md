# Data sources, quirks, and units

All checked live on 2026-11-01. Paths are under `data/` (git-ignored).

| Source | Code | Cached at | Notes |
|---|---|---|---|
| AirNow HourlyAQObs | `data/airnow.py` | `raw/airnow/hourly/YYYYMMDD.parquet` | UTC, hour-beginning. Has lat/lon + reporting areas. Since 2019-05-28. Small negatives (>-5) are valid. |
| AirNow reportingarea.dat | `data/airnow.py` | `raw/airnow/reportingarea/` | Last file of each day. F = forecast (in force that day; may have been issued earlier), Y = yesterday's observed area AQI, O = hourly obs. Forecasts submitted late morning-afternoon local, all by 22Z. Some areas give only a category, no AQI number. |
| HMS smoke | `data/hms.py` | `raw/hms/*.zip` | Polygons with Density Light/Medium/Heavy (analyst estimate of column smoke, roughly 0-10 / 10-21 / 21-32 µg/m³). Column ≠ ground level. |
| HRRR (sfc) | `data/hrrr.py` | `interim/hrrr/YYYYMMDD_12z.parquet` | Byte-range downloads via `.idx`. MASSDEN 8 m (near-surface smoke) every hour f00-f48; met + COLMD every 3 h. MASSDEN/COLMD units: detected by magnitude (the label changed 2021-12-21). Stored in µg/m³ and mg/m². |
| AQM (NAQFC) | `data/aqm.py` | `interim/aqm_bc/YYYYMMDD_12z.parquet` | Hourly 1-h averages, 72 steps. v5 → 2021-07-19, v6 → 2024-05-13, v7 after. Same filename pattern in all versions. The `ave_24hr` files use a non-local-day window, so we don't use them. 9999 = fill. |
| FIRMS | `data/firms.py` | `raw/firms/` | Needs free MAP_KEY. VIIRS SNPP + NOAA-20 standard processing for past seasons. |
| PurpleAir | `data/purpleair.py` | `raw/purpleair/` | Needs key + paid points. A/B QC, EPA correction (Barkjohn 2021; 2022 smoke extension). |

## Local days
The AQI daily value is midnight-to-midnight local *standard* time. Each site's
`gmt_offset` comes from AirNow (Pacific = -8). Grid points: -8 west of -114.5°,
else -7.

## The training table (`processed/<study>/table_<split>.parquet`)
One row per (issue_date, point_id). Columns are described at the top of
`serein/features/table.py`. TODO A-15: full data dictionary with units and
missing-value rates.

## Size
Five seasons: AirNow ≈ 20 GB downloaded → ~150 MB cached; HRRR ≈ 90 GB
downloaded → a few GB of point values; AQM ≈ 45 GB downloaded → ~1 GB.
