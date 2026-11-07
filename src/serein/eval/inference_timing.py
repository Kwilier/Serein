"""Measure local calibrated model inference without evaluating forecast targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from serein.features.graph import Graph, build_knn_graph
from serein.features.prepare import GraphData, target_cols, to_graph_data
from serein.models.conformal import calibrate_quantiles
from serein.models.gnn import QUANTILES
from serein.models.infer import load_artifacts


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_forward(model, data: GraphData, graph: Graph, cqr, *,
                      device: str = "cpu", repeats: int = 10, warmup: int = 2):
    """Return timing statistics and the last calibrated PM quantiles.

    The supplied model is moved to device and put in evaluation mode. Data must
    contain one issue day. Targets are never used. Peak CUDA tensor allocation
    includes the resident model and inputs, not just incremental activations.
    """
    for name, value, minimum in (("repeats", repeats, 1), ("warmup", warmup, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if len(data.days) != 1 or data.X.ndim != 3 or data.X.shape[0] != 1 or not len(data.nodes):
        raise ValueError("timing requires one nonempty issue day")
    if getattr(data, "hist", None) is not None:
        raise ValueError("history inputs require a separate temporal-model timing path")
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    def synchronize():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    start = time.perf_counter()
    model.to(dev).eval()
    arguments = (
        torch.as_tensor(data.X, dtype=torch.float32, device=dev),
        torch.as_tensor(data.wind, dtype=torch.float32, device=dev),
        torch.as_tensor(data.base, dtype=torch.float32, device=dev),
        torch.as_tensor(graph.src, dtype=torch.long, device=dev),
        torch.as_tensor(graph.dst, dtype=torch.long, device=dev),
        torch.as_tensor(graph.bearing, dtype=torch.float32, device=dev),
        torch.as_tensor(graph.edge_attr, dtype=torch.float32, device=dev),
    )
    synchronize()
    setup_seconds = time.perf_counter() - start
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    @torch.inference_mode()
    def forward():
        q = model(*arguments).detach().cpu().numpy()
        if q.shape != (1, len(data.nodes), data.base.shape[-1], len(QUANTILES)):
            raise ValueError("model output does not match the daily forecast shape")
        calibrated = calibrate_quantiles(q[0], cqr, QUANTILES)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.expm1(calibrated)
        if not np.isfinite(values).all() or (values < 0).any() or (np.diff(values, axis=-1) < 0).any():
            raise ValueError("inference produced invalid calibrated PM quantiles")
        return values

    synchronize()
    start = time.perf_counter()
    first = forward()
    synchronize()
    first_seconds = time.perf_counter() - start
    for _ in range(warmup):
        forward()
    samples = []
    largest_difference = 0.
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        values = forward()
        synchronize()
        samples.append(time.perf_counter() - start)
        largest_difference = max(largest_difference, float(np.max(np.abs(values - first))))
    report = {
        "device": str(dev), "device_name": (torch.cuda.get_device_name(dev)
                                               if dev.type == "cuda" else platform.processor()),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "python_version": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "nodes": len(data.nodes), "edges": graph.n_edges, "features": data.X.shape[-1],
        "leads": data.base.shape[-1], "ensemble_members": len(getattr(model, "members", [model])),
        "repeats": repeats, "warmup": warmup, "device_setup_seconds": setup_seconds,
        "first_calibrated_forward_seconds": first_seconds,
        "calibrated_forward_seconds": samples,
        "median_seconds": float(np.median(samples)), "p90_seconds": float(np.quantile(samples, .9)),
        "max_abs_repeat_difference": largest_difference,
        "cuda_peak_allocated_bytes": (int(torch.cuda.max_memory_allocated(dev))
                                        if dev.type == "cuda" else None),
        "quantiles_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "scope": "resident model, device-to-host output, calibration and expm1; no target scoring",
    }
    return report, values


def profile_export(run: Path, table: pd.DataFrame, points: pd.DataFrame, issue_date,
                   *, device="cpu", repeats=10, warmup=2, neighbors=10, max_km=300):
    """Prepare one day and profile a trusted, locally exported artifact bundle."""
    start = time.perf_counter()
    model, prep, cqr, metadata = load_artifacts(run)
    load_seconds = time.perf_counter() - start
    if metadata.get("history", 1) != 1:
        raise ValueError("history models require temporal input preparation")
    start = time.perf_counter()
    day = pd.Timestamp(issue_date).date()
    labels = pd.to_datetime(table.issue_date, format="mixed").dt.date
    selected = table[labels.eq(day)].copy()
    if selected.empty:
        raise ValueError("issue date has no input rows")
    if selected.point_id.isna().any() or selected.point_id.duplicated().any():
        raise ValueError("daily point IDs must be present and unique")
    if points.point_id.isna().any() or points.point_id.duplicated().any():
        raise ValueError("point coordinates require present, unique IDs")
    nodes = metadata["node_ids"]
    indexed = points.set_index("point_id")
    if not set(nodes).issubset(indexed.index):
        raise ValueError("coordinates are missing model nodes")
    indexed = indexed.loc[nodes]
    graph = build_knn_graph(indexed.lat, indexed.lon, k=neighbors, max_km=max_km)
    # Preparation needs lead columns, but timing must not use observed outcomes.
    selected = selected.drop(columns=target_cols(selected))
    selected["issue_date"] = day
    for lead in metadata["leads"]:
        selected[f"y_L{lead}"] = np.nan
    data = to_graph_data(selected, prep, nodes)
    prepare_seconds = time.perf_counter() - start
    report, values = benchmark_forward(model, data, graph, cqr, device=device,
                                       repeats=repeats, warmup=warmup)
    report.update({"artifact_load_seconds": load_seconds, "prepare_seconds": prepare_seconds,
                   "issue_date": day.isoformat(), "input_rows": len(selected),
                   "present_model_nodes": int((data.row_index >= 0).sum()),
                   "neighbors": neighbors, "max_km": max_km,
                   "artifact_format": metadata.get("format_version", 1)})
    return report, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--issue-date", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--max-km", type=float, default=300.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output already exists; choose a new path")
    report, _ = profile_export(args.run, pd.read_parquet(args.table), pd.read_parquet(args.points),
                               args.issue_date, device=args.device, repeats=args.repeats,
                               warmup=args.warmup, neighbors=args.neighbors, max_km=args.max_km)
    paths = {"table": args.table, "points": args.points,
             **{name: args.run / name for name in ("model.json", "smokecast.pt", "artifacts.pkl")}}
    report["inputs"] = {name: {"path": str(path.resolve()), "sha256": _file_hash(path)}
                        for name, path in paths.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
