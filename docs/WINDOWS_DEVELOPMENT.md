# Running the development study on Windows

Use the repository's Python 3.12 environment from PowerShell. The native runner
does not need Bash, WSL, or a separate installation of `uv` inside a Linux shell.
The existing shell scripts remain available for Linux and macOS.

From the repository root:

```powershell
uv sync --all-extras
uv run python -m serein.eval.development plan --tag dev_windows
```

`plan` prints the exact commands and output paths without creating directories or
starting subprocesses. It uses the active Python interpreter for every step, so a
Windows virtual environment stays in use throughout the study.

The runner expects the study's public environmental data cache, point registry,
and rebuilt training and calibration tables to be available. It does not download
data or rebuild those tables. Follow the data preparation steps in
[NEXT_STEPS.md](NEXT_STEPS.md) first, especially after observation aggregation or
missing-data handling changes. Cached development results from an older feature
table do not validate the revised data pipeline.

## Check the compute runtime before a long fit

If a test or fit hangs, first run a bounded check with synthetic tensors:

```powershell
New-Item -ItemType Directory -Path outputs -Force | Out-Null
uv run python -m serein.eval.runtime_check --output outputs/runtime_cpu_probe
```

Use `--device cuda` in a different new output directory to check GPU allocation
and gradients. `--timeout-seconds` changes the default 30-second limit. This
command loads no environmental data, fitted weights, or held-out observations.
It uses one intra-operation and one inter-operation CPU thread inside its fresh
worker, without changing the parent shell or installed packages.

The log records import, device allocation, forward, and backward checkpoints.
`manifest.json` retains the PyTorch/Python versions, requested device, last
checkpoint, and status. Exit 0 requires the full sequence and finite nonzero input
gradients. Timeouts return 124 after process-tree cleanup; child failures keep
their exit code, launch/cleanup errors return 127, and interruption returns 130.
Existing output directories are refused.

A timeout at `backward` means imports and a simple linear forward pass completed
but its gradient did not finish within the limit. This check does not import a
Serein model, so it helps separate a runtime problem from a graph-model problem.
An allocation timeout can instead indicate slow or blocked device initialization.
Neither result identifies the underlying driver, package, security-policy, or
resource issue by itself. Preserve the report before investigating the environment.
A passing probe checks basic execution, not the full pipeline or forecast skill.

## Starting a run

After reviewing the plan:

```powershell
uv run python -m serein.eval.development run --tag dev_windows
```

Choose a new tag for each run. Without `--tag`, the runner generates a timestamped
tag. The full suite performs the following work in order:

1. Fit and score the development backtest on calibration-season scoring weeks.
2. Generate figures and the reporting-area warning impact report.
3. Export the [paired daily-loss comparison](FORECAST_COMPARISON.md) and its audit.
4. Run the development monitor holdout experiment with 20% of sites hidden.
5. Run the full-input ablation and each of the six input-group ablations.

The primary development and holdout runs use five ensemble members. Each input
ablation uses three, matching the existing shell workflow. Within a backtest, the
graph/no-graph comparison uses matching ensemble sizes. Default training is 300
epochs. This is a study rerun, so allow time for several model fits.

To bound the entire suite, add `--timeout-minutes 60`. The time limit is shared
across steps. When it expires, the active Python process tree is stopped, later
steps are omitted, and the manifest records `timed_out` with exit code 124. Logs
and partial artifacts are retained. Without this option there is no time limit.

The runner exposes no reserved-evaluation mode, transfer command, arbitrary
backtest-argument passthrough, or overwrite option. It never loads the test table
through its generated backtest commands. This does not replace the team's review
and protocol-freezing process.

To run only part of the suite:

```powershell
uv run python -m serein.eval.development run --suite dev
uv run python -m serein.eval.development run --suite ablations
```

`dev` includes reports and the monitor holdout run. `ablations` includes all seven
input-group runs. `--config` selects another study configuration. `--epochs`,
`--seeds`, and `--ablation-seeds` allow an explicitly different runtime check, but
changing them does not reproduce the default study. In particular, a short run
only checks that the pipeline executes; it provides no replacement accuracy claim.

## Optional map

Maps are opt-in because the requested date must have the needed environmental
inputs cached. The date must fall in the configuration's calibration years and
season months:

```powershell
uv run python -m serein.eval.development plan --suite dev `
  --tag dev_windows_map --map-date 2023-08-18
uv run python -m serein.eval.development run --suite dev `
  --tag dev_windows_map --map-date 2023-08-18
```

The forecast parquet and map image go in the new development run's `maps/`
subdirectory. They do not replace the shared `outputs/maps/` images. To render an
already saved development model separately, choose a new output directory:

```powershell
uv run python -m serein.eval.development map `
  --config configs/west_us.yaml --date 2023-08-18 `
  --run outputs/backtest/west_us_dev_windows `
  --output outputs/backtest/west_us_dev_windows/map_review
```

## Outputs and failures

For study `west_us` and tag `dev_windows`, the main model lives under
`outputs/backtest/west_us_dev_windows/`, the site holdout under
`outputs/backtest/west_us_dev_windows_holdout/`, and the input ablations under
`outputs/backtest/west_us_dev_windows_abl_*/`.

Logs and comparison CSVs live under `outputs/logs/west_us_dev_windows/`. Each step
has its own UTF-8 log, including the exact argument vector and merged stdout and
stderr. `impact.log` contains the impact report. The manifest records commands,
UTC start and finish times, completed steps, and the exit status. A failure or
keyboard interruption retains the logs and a partial-progress manifest.

Each step owns its worker processes for its whole lifetime. On Windows 10 or
newer, the runner creates the child suspended inside a private
[job object](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects),
then starts it. Job membership uses the documented
[process-creation attributes](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute), so
interrupting the owner cannot leave a child between creation and assignment.
This also contains children of the virtual-environment launcher.
The job is stopped on timeout, interruption, launch failure after creation, and
normal completion, so a finished parent cannot leave background workers running.
The runner verifies that the job has no active processes before continuing. A
cleanup failure is reported as an error rather than successful completion.

The Windows path uses built-in APIs and requires no administrator access or
process-list enumeration. Only the child's log and empty standard input are
inherited; unrelated inheritable handles are excluded. Each log includes the
owned process ID for diagnosis. On POSIX systems, a separate process session
provides process-group cleanup. Commands run without a shell, with empty standard
input. These steps must not launch independent long-lived services.

Existing output directories are refused before any subprocess starts, even if
only a later ablation directory already exists. Run directories are reserved at
startup, so a failed run can leave empty directories for steps not reached. Keep
the failed run for diagnosis and use a new tag to retry. There is no resume or
automatic cleanup.

The first nonzero child-process exit stops the suite and becomes the runner's
exit code. Later steps do not run, and `ALLDONE` is printed only after every
requested step succeeds. A successful process exit checks execution; review the
figures, coverage, warning tradeoffs, and comparison populations separately.

## Verification

Before interpreting development scores, run the [target-label split audit](SPLIT_AUDIT.md).
Different issue weeks can reuse the same observed target day at different leads.
The audit proposes calendar exclusions for review without changing the backtest.

```powershell
uv run pytest tests/test_development.py tests/test_process.py tests/test_comparison.py -q
uv run ruff check .
uv run pytest -q
```

If pytest reports `WinError 5` while opening an existing `pytest-of-*` directory,
give this session a new parent under Windows Temp, then rerun the same checks:

```powershell
$env:PYTEST_DEBUG_TEMPROOT = Join-Path ([IO.Path]::GetTempPath()) `
  ('serein-pytest-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $env:PYTEST_DEBUG_TEMPROOT | Out-Null
uv run pytest tests/test_development.py tests/test_process.py tests/test_comparison.py -q
```

This setting applies to the current shell and its children. Pytest still creates
its normal per-test folders inside that parent. Keep scratch data in Windows
Temp; do not point pytest's base directory at the repository or `outputs/`.

The native-runner tests launch real small Python subprocesses and check success,
failures at each position, merged logs, preserved outputs, cancellation, and paths
with spaces. Process tests check descendant cleanup after timeout, interruption,
and parent completion, job-configuration failure, Unicode argument round trips,
stable Windows handle counts, and an unrelated process left running. Map tests use
a synthetic forecast. The existing shell-script tests
still require a working POSIX Bash environment; the Windows app-execution alias
for `bash.exe` alone does not provide one. A shell-test launch failure must be
reported separately from the native runner's results. These tests and short
runtime probes do not establish forecasting accuracy or complete the final study.
