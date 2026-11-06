# Neighborhood grids and tract summaries

`python -m serein.spatial` prepares kilometer-spaced map cells, exports development
forecasts from a prepared grid, draws the cell footprints, and summarizes forecasts
over local tract boundaries. It reads local files and writes new outputs. It does
not download boundaries, train models, score observations, or run the reserved
evaluation. Data preparation uses the existing download commands separately.

This implements B-9b's map and tract export path. A smaller map cell does not
establish a corresponding increase in forecast resolution or skill. The uncertainty
calibration at new grid locations still needs spatial validation.

## Summarize an existing development forecast

Use the original point registry and the spacing used to create it. The default
western-US study uses a degree grid. Supply both spacings explicitly; the tool
does not infer larger cells from gaps in a sparse registry.

```powershell
uv run python -m serein.spatial tracts `
  --forecasts app/sample/forecast_20230818.parquet `
  --points data/interim/west_us/points.parquet `
  --lat-step 0.25 --lon-step 0.25 `
  --tracts local_tracts.geojson --tract-id GEOID `
  --output tract_forecast.parquet
```

The output's parent directory must already exist. Existing outputs are refused.
The registry must correspond to the saved forecast. Grid forecasts with unknown
point IDs fail rather than being dropped. Sensor forecasts are excluded. If the
forecast includes coordinates, each point must lie in its assigned cell.

The tract file can be a local GeoJSON, shapefile, or GeoParquet file. It must have
a CRS, valid polygon or multipolygon geometry, and unique nonempty string IDs.
Keep Census GEOIDs as strings to preserve leading zeroes. The default ID column
is `GEOID`; `--tract-id` supports other polygon datasets.

Choose and retain a boundary vintage. Census [cartographic boundary files](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html)
are simplified boundaries. Their simplification and treatment of coastlines
affect area weights, especially in small or coastal tracts. This tool uses the
supplied geometry as its area denominator, including any water it contains. It
does not infer land-only or population-weighted exposure.

## Prepare a finer metro grid

Bounds are west, south, east, north in longitude/latitude. This example prepares a
Sacramento-area grid without changing the existing study registry:

```powershell
uv run python -m serein.spatial grid `
  --bounds -121.7 38.3 -121.2 38.8 --spacing-km 1 `
  --sites data/interim/west_us/points.parquet `
  --output sacramento_cells.parquet `
  --registry-output sacramento_points.parquet
```

`--spacing-km 2` makes a coarser alternative. The grid uses EPSG:5070, NAD83 / Conus
Albers. Distances are in projected kilometers, not an assertion that every cell
edge has that exact ground distance. Its equal-area squares are anchored at the
projection origin, so overlapping requests at the same spacing share cell IDs
and centers. Edge cells remain whole and can extend outside the requested box.
The bounding box is densified before projection to retain curved projected edges.

The supported bounds are inside `(-125, 24, -66, 50)`. Alaska, Hawaii, territories,
and dateline-crossing grids need a different projection and are not supported by
this generator. The default allocation cap is 100,000 candidate cells. The tool
checks it before allocating cell arrays; `--max-cells` sets a different cap.

The cells file includes geometry, `point_id`, `kind=grid`, center `lat`/`lon`,
`gmt_offset`, and `spacing_km`. With `--sites`, time zones come from the nearest EPA
monitor, excluding old grid rows. Without usable monitor offsets, the existing
longitude-based standard-time fallback applies. Inspect time-zone borders before
using the registry for daily model inputs.

`--registry-output` requires `--sites`. It exports the supplied registry's sensor
rows with their metadata unchanged, removes its old grid rows, and appends the
new cells' center rows. It never rewrites the source registry. The geometry stays
in the separate cells file. In Python:

```python
import geopandas as gpd
import pandas as pd

cells = gpd.read_parquet("sacramento_cells.parquet")
points = pd.read_parquet("sacramento_points.parquet")
assert set(points.loc[points.kind == "grid", "point_id"]) == set(cells.point_id)
```

This export is preparation for a separate development study. It does not install
the registry into the main study, preserve a study's exclusion boxes automatically,
or resample the existing NOAA cache. Before using it with the model:

1. Keep the trained monitor IDs and their metadata. Apply the study's region and
   exclusion rules to the new grid, preserving matching cells and center rows.
2. Use a separate study name and cache/output paths. Extract HRRR/AQM inputs for
   the new point IDs with the same issue-time rules. Existing coarse-grid cache
   rows do not provide forecasts at these new IDs.
3. Check the daily inputs, then run development inference with the reviewed trained
   ensemble. Inspect missing inputs and sensor-neighbor distances. Keep the original
   cells file beside that forecast for aggregation.

Use the [map support audit](MAP_SUPPORT.md) to inspect structural sensor support
without rerunning inference. It records disconnected grid nodes and missing trained
sensors without treating graph connectivity as forecast skill.

## Run development inference on the new registry

Use a separate named study and cache so a new grid cannot invalidate the main
study's extracted NOAA inputs. Starting with the two Sacramento files above, this
PowerShell example creates new paths and preserves the original configuration:

```powershell
@'
from pathlib import Path
import shutil
import yaml

study = yaml.safe_load(Path("configs/west_us.yaml").read_text(encoding="utf-8"))
study["name"] = "west_us_sacramento"
with Path("configs/west_us_sacramento.yaml").open("x", encoding="utf-8") as stream:
    yaml.safe_dump(study, stream)
registry = Path("data/neighborhood_sacramento/interim/west_us_sacramento/points.parquet")
registry.parent.mkdir(parents=True, exist_ok=False)
shutil.copy2("sacramento_points.parquet", registry)
'@ | uv run python -
```

Populate that cache with the environmental inputs for a development issue date.
The observation padding below covers the default model's preceding days and the
local-day targets used by table construction. HMS includes the preceding day's
smoke polygons. The NOAA commands extract the new point IDs rather than reusing
the old grid's sampled rows. If these environmental files are already available
locally, they can instead be copied into the equivalent new cache paths.

```powershell
$neighborhoodDataBefore = $env:SEREIN_DATA
$env:SEREIN_DATA = (Resolve-Path data/neighborhood_sacramento).Path
try {
  uv run serein download airnow --config configs/west_us_sacramento.yaml `
    --start 2023-08-15 --end 2023-08-22
  uv run serein download hms --config configs/west_us_sacramento.yaml `
    --start 2023-08-17 --end 2023-08-18
  uv run serein download hrrr --config configs/west_us_sacramento.yaml `
    --start 2023-08-18 --end 2023-08-18 --workers 4
  uv run serein download aqm --config configs/west_us_sacramento.yaml `
    --start 2023-08-18 --end 2023-08-18 --workers 4
  uv run python -m serein.spatial forecast `
    --config configs/west_us_sacramento.yaml --date 2023-08-18 `
    --run outputs/backtest/west_us_reviewed_dev --cells sacramento_cells.parquet `
    --output sacramento_export
} finally {
  $env:SEREIN_DATA = $neighborhoodDataBefore
}
```

Replace the model directory with a reviewed development export that actually
exists. The forecast command loads its local weights and preprocessing pickle, so
use trusted model artifacts. It uses every exported ensemble member. For a model
with history greater than one day, extend the NOAA and HMS date ranges to cover
those earlier input days and pad observations accordingly.

The forecast command checks that the date belongs to the configuration's
calibration season, that the registry's grid IDs match the cells, that centers
obey the study region and exclusions, and that both NOAA caches contain every
grid ID for each required history day. Their point-request hashes must match the
current registry, including its coordinates. These checks prevent an untouched
coarse-grid cache from silently becoming imputed inputs at new locations. They
do not establish completeness of every field or hour. Inspect the feature tables
and extraction completeness metadata as well.

The new directory contains `forecast.parquet`, `cells.parquet`, `map.png`,
`grid_support.parquet`, and `summary.json`. The summary records the member count,
seeds, history length, point-request hash, source paths and structural support.
An error can leave a partial new export directory; use a new destination on retry.
An existing destination is refused.

The renderer draws polygon footprints directly. The existing `serein map` renderer
assumes a regular degree grid and is unsuitable for projected kilometer cells.
To redraw a saved cell forecast without loading a model:

```powershell
uv run python -m serein.spatial plot `
  --forecasts sacramento_export/forecast.parquet --cells sacramento_export/cells.parquet `
  --output sacramento_redrawn.png
```

Panels share a continuous PM2.5 color scale. Missing cell forecasts are gray and
hatched. The default panel cap is 12; `--max-panels` can change it explicitly.
Subdivide large multi-day exports before rendering them.

With a saved forecast at the projected cells:

```powershell
uv run python -m serein.spatial tracts `
  --forecasts sacramento_export/forecast.parquet --cells sacramento_export/cells.parquet `
  --tracts local_tracts.geojson --output sacramento_tracts.parquet
```

## What a tract value means

For each issue date and lead, the tool intersects grid-cell footprints with each
tract in EPSG:5070. Only positive intersection areas contribute. A cell's forecast
is treated as constant within its footprint. Cell overlaps are rejected so area
cannot be counted twice; shared edges are allowed. Holes and disconnected tract
parts retain their geometry.

`q50_area_mean` is the intersection-area-weighted mean of available grid-cell
medians, in micrograms per cubic meter. It is not the median of the tract's joint
forecast distribution. Pointwise quantile bounds cannot establish a calibrated
interval for a spatial average, so this tool does not aggregate them or assign a
tract AQI category. It also does not replace EPA reporting-area comparisons, which
use a different warning rule.

Missing forecast rows and missing medians reduce coverage. Negative or infinite
medians, duplicate cell/date/lead forecasts, inconsistent valid dates, invalid
geometry, and overlapping cells fail before writing an output. The input date
columns are timezone-free calendar labels, and `valid_date` must equal
`issue_date + lead` days. Each represented issue date and lead gets a row for
every supplied tract, even when no cell reaches that tract. An issue date and
lead absent from the forecast entirely cannot be inferred and has no output rows.

| Column | Meaning |
|---|---|
| `tract_id` | Original polygon identifier, retained as a string |
| `issue_date`, `lead`, `valid_date` | Forecast group |
| `q50_area_mean` | Weighted mean of cell medians, blank unless coverage passes |
| `tract_area_m2` | Full supplied tract polygon area in the equal-area projection |
| `grid_area_m2`, `grid_coverage` | Tract area intersected by the grid, regardless of forecast availability |
| `available_area_m2`, `forecast_coverage` | Tract area intersected by cells with a finite median in this forecast group |
| `n_cells`, `n_available` | Intersecting cells and cells with an available median |
| `min_coverage` | Threshold used to publish the average |
| `status` | `ok`, `insufficient_coverage`, or `no_forecast` |
| `geometry` | Original tract polygon in its input CRS |

Coverage fractions use the full tract area, not only the part inside the grid.
The default minimum forecast coverage is 0.95. Below that threshold, the average
is blank and `status=insufficient_coverage`; with no available area,
`status=no_forecast`. `--min-coverage` can lower the threshold explicitly for an
exploratory map. A published value below full coverage averages only the available
area and can still be biased by the missing part. Preserve the coverage columns.

Projection follows GeoPandas [coordinate transformation rules](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html):
segments connect vertices in the source CRS. Geometry precision and boundary
detail limit the accuracy of the resulting area calculation. Summaries require
CONUS-appropriate boundaries and do not repair invalid shapes automatically.

Read or export the result using GeoPandas:

```python
import geopandas as gpd

summary = gpd.read_parquet("tract_forecast.parquet")
print(summary[["tract_id", "lead", "q50_area_mean", "forecast_coverage", "status"]])
```

Validation uses synthetic polygons and public environmental data. Tests check
partial cells, holes, disconnected polygons, missing forecasts, projection
roundtrips, analytical rectangle weights, stable grid IDs, file preservation,
development-date restrictions, and inference with real two-member synthetic
artifacts. Shared point forecasts stay unchanged when other grid nodes are added.

On 2026-11-06, isolated caches were used to extract fresh HRRR/AQM inputs for
2023-08-18 and run a newly exported five-member development ensemble at 2,542
Sacramento and 2,735 Portland kilometer-grid points. All exported quantiles were
finite, nonnegative and ordered. Shared EPA forecasts were identical across the
two maps. Census 2023 tract summaries covered 363 Sacramento County tracts and
197 Multnomah County tracts, with incomplete or absent coverage retained explicitly.
This establishes execution of the export path. It does not establish neighborhood
forecast accuracy, spatial uncertainty coverage, or clinical suitability. The
training used existing cached development tables, which were not rebuilt here.
