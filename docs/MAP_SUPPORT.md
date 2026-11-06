# Auditing sensor support in saved maps

A finite map forecast can exist at a location with no nearby trained sensor.
The sensor graph uses a configured neighbor count and distance cutoff; it does
not establish that a sensor has a usable reading on that issue day. Inspect this
support separately from model accuracy.

```powershell
uv run python -m serein.eval.map_support `
  --config configs/west_us.yaml `
  --model-json outputs/backtest/west_us_dev/model.json `
  --points data/interim/west_us/points.parquet `
  --forecasts app/sample/forecast_20230818.parquet `
  --output map_support_audit
```

Choose the registry, graph configuration, and model metadata that were actually
used for that saved forecast. The tool reconstructs structural support from these
supplied files. It cannot prove they describe a historical export's original
configuration. Forecast coordinates, when included, must match the registry within
one millionth of a degree. Unknown grid IDs and duplicate point/date/lead keys fail
before creating the report. The output directory must be new.

The command reads `model.json` only. It does not load model weights or pickle
files, access the network, or run inference or an evaluation. By default it does
not read observations. An optional prepared feature table adds reading availability.
The plot uses no downloaded basemap. All entries are environmental data.

## Outputs

| File | Contents |
|---|---|
| `grid_support.parquet` | Every grid point in the registry, with neighbor counts and distances |
| `forecast_audit.parquet` | Saved grid forecasts joined to their support, retaining forecast values |
| `forecast_counts.csv` | Counts of forecast rows without sensor edges or valid medians, by issue date and lead |
| `summary.json` | Graph settings, available and missing trained sensors, registry-level counts, source paths and SHA-256 digests |
| `support.png` | Grid neighbor counts and nearest trained-sensor distances, with disconnected points marked |

## Check recorded neighbor readings

Add `--features data/processed/west_us/table_calibrate.parquet` to the command to
inspect the model's prepared as-of observation features for each forecast date.
This table must correspond to the study, registry, issue-time rules, and dates
being audited. Its target columns are ignored.

The forecast audit then includes:

| Column | Meaning |
|---|---|
| `n_neighbors_with_feature_row` | Incoming sensor neighbors with a feature-table record on this issue date |
| `n_neighbors_with_last24` | Incoming neighbors with a finite `obs_last24` |
| `n_neighbors_with_prevday` | Incoming neighbors with a finite `obs_prevday` |
| `reading_support` | `last24_available`, `no_last24_readings`, `no_feature_records`, or `no_sensor_edges` |

A missing feature record is distinct from an existing row with a missing reading.
Untrained sensors cannot supply readings to the audited graph. Infinite values are
unavailable; small negative readings remain finite records, consistent with the
observation pipeline's handling of measurement noise. These are availability counts,
not a new QC screen. One finite reading does not establish a well-observed network.
The CSV adds counts of forecast rows with no recorded neighbor, no finite 24-hour
neighbor reading, or no finite previous-day neighbor reading. These counts can
overlap, including at points with no edges. The plot remains a structural map.

The tool checks calendar-date keys and rejects duplicate feature point/date rows.
It cannot establish that an arbitrary supplied table obeys the no-peeking rules,
contains sufficient observation hours, or passed sensor QC. Use the prepared
feature pipeline and retain its provenance. This optional report describes the
provided records; it does not reconstruct historical readings from current data.

Available sensors are the model's `node_ids` present in the registry, in the
model's order. An extra sensor in the registry cannot supply support just because
it happens to be closer. Missing trained sensors are retained in the summary's
`missing_trained_sensor_ids` list. A trained sensor ID referring to a grid row is
an error.

The graph uses the same `sensor_plus_grid_graph` function as map inference, so
sparse networks use the available neighbor count and enforce the supplied distance
cutoff. `n_sensor_neighbors` counts the incoming sensor edges at a grid node.
`closest_used_sensor_km` and `farthest_used_sensor_km` describe those selected edges.
They are blank when there are no edges. `nearest_trained_sensor_km` searches all
available trained sensors, even beyond the cutoff, so a disconnected cell still
has a distance when any trained sensor is available. With no available sensors,
its distance is blank and the cell has zero edges.

The grid report covers the entire supplied registry. The forecast audit and CSV
cover only grid rows represented in the forecast, excluding sensor forecast rows.
An absent forecast date or point is not invented. `valid_median` is true only for
a finite nonnegative `q50`. Invalid and missing medians remain in the audit and
are counted rather than filled or discarded.

## Interpretation

`sensor_support=connected` means an edge is available in the reconstructed graph.
It does not mean that the sensor's current reading was available, accepted by QC,
or influential in the model. For graph-disabled models, the map's sensor-neighbor
base can still use the graph; this audit does not measure the learned correction's
dependence on those edges.

`sensor_support=no_sensor_edges` marks a location that cannot receive a sensor
edge under the supplied settings. A finite forecast there may still come from
NOAA inputs or the model's fallback. It should not be described as a locally
sensor-supported prediction. Neither a large neighbor count nor a short distance
establishes forecast accuracy or spatial resolution.

Use this report when interpreting neighborhood maps. Keep it alongside the forecast
and any tract coverage report. Sensor support describes the graph; tract coverage
describes the area represented by available grid forecasts. They answer different
questions.

Tests use synthetic sensors, sparse and missing networks, a dateline case, and
file roundtrips. The command can also inspect saved development maps without
changing their predictions or results.
