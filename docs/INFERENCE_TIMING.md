# Measure local inference before estimating hosting cost

`serein.eval.inference_timing` measures an exported model on one issue day's
already-built inputs. It does not fit a model, score targets, fetch data, run a
backtest or invoke the reserved evaluation. This provides one measured input to
the [pilot budget worksheet](BUSINESS_MODEL_EVIDENCE.md), not a complete service
cost or an accuracy result.

## Profile an existing development export

Use a trusted local artifact bundle, the matching development table and point
registry, and the graph settings used for that model:

```powershell
uv run python -m serein.eval.inference_timing `
  --run outputs/backtest/west_us_dev `
  --table data/processed/west_us/table_calibrate.parquet `
  --points data/interim/west_us/points.parquet --issue-date 2023-08-18 `
  --neighbors 10 --max-km 300 --device cpu --repeats 20 --warmup 3 `
  --out outputs/timing/development_cpu.json
```

Use `--device cuda` or a specific CUDA device, such as `cuda:0`, for a separate
GPU profile and choose a new output path. A requested CUDA device fails if
unavailable; the tool does not fall back to CPU. Existing output files are
refused. No original artifact, input table or registry is written.

The saved model's node IDs determine the graph and node order. The registry must
provide every model node's coordinates. Input rows may be missing for some
nodes; the report records the number actually present. Input rows with duplicate
or missing point IDs are rejected. The CLI profiles the trained-node graph,
not an expanded neighborhood map. Temporal models requiring input history need
a separate preparation path and are rejected.

Legacy single-member exports remain single-member timing cases, including
exports whose directory name mentions an ensemble. Read `artifact_format` and
`ensemble_members` in the report rather than inferring member count from the
path. A current ensemble is loaded and timed with all exported members.

## What the report measures

| Field | Scope |
| --- | --- |
| `artifact_load_seconds` | Loading model weights, metadata and preprocessing/calibration into the current process |
| `prepare_seconds` | Selecting the issue day, arranging nodes, building its graph and preparing numeric inputs |
| `device_setup_seconds` | Moving the model and input tensors to the requested device, including synchronization |
| `first_calibrated_forward_seconds` | First forward call, output transfer to CPU, calibration and conversion from log1p PM |
| `calibrated_forward_seconds` | Individual timed calls after the first call and the requested additional warmup calls |
| `median_seconds`, `p90_seconds` | Summaries of those timed samples, without a concurrency or service-level guarantee |
| `max_abs_repeat_difference` | Largest difference in calibrated PM quantiles between the first and timed calls |
| `cuda_peak_allocated_bytes` | Peak process-local tensor allocation since device setup; null for CPU |

The report includes versions, CPU thread count, device identity, node/edge/feature
counts, forecast leads, member count, graph settings, input file hashes and a
hash of the final quantile array. Quantiles must be finite, ordered and
nonnegative or the profile fails. Original target columns are discarded and
replaced with missing placeholders before input preparation; changing their
values does not change the timed forecast. No predictive metrics are computed.

GPU kernels run asynchronously. Each timed call starts and ends with a
[CUDA synchronization](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.synchronize.html)
when using CUDA. Inference runs with
[inference mode](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.inference_mode.html)
and the model in evaluation mode. The peak tensor metric uses
[reset_peak_memory_stats](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.reset_peak_memory_stats.html)
and [max_memory_allocated](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html).
These API references were checked for the locally installed PyTorch 2.11.

The CUDA peak includes resident model/input tensors and allocations from the
first call, warmups and timed calls. It can also include other tensor allocations
in the same process. It excludes allocator reservations, driver/context memory
and other processes. It is not total GPU memory use. CPU peak memory is not
measured; null must not be treated as zero.

## Use the measurement within its limits

Run on a quiet, representative deployment and retain every timing sample. Report
the actual hardware, software, input sizes and member count. Check CPU/GPU output
agreement separately if changing device. Warmed repeated calls on one graph are
not independent production requests or a measurement of throughput under load.

The CLI reads table and registry files before the preparation timer. Disk reads
of those inputs, raw weather decoding, downloads, feature-table construction,
virtual grid generation, map rendering, web requests, startup outside artifact
loading, monitoring, retries and concurrent users are excluded. Artifact load
and device setup are measured within the current process, not as a fully cold
OS, disk-cache or container startup.

For a serverless deployment, also measure process/container startup and determine
which work repeats per request. For a resident service, measure capacity and
concurrency rather than multiplying a single call by requests. Apply a provider
rate only to a comparable deployment and billing unit. A fast desktop forward
pass does not establish a cloud invoice, availability guarantee or full daily
forecast runtime.

For prepared map graphs or another caller, `benchmark_forward(model, data,
graph, cqr, device=..., repeats=..., warmup=...)` returns a report and its last
calibrated quantile array. It moves the supplied model and puts it in evaluation
mode, so use an inference-owned model instance. `profile_export(...)` adds
artifact loading and daily input preparation; the CLI adds input-file hashes.
