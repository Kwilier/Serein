"""Audit sensor support for saved map forecasts without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from serein.config import load_study
from serein.data.firms import haversine_km
from serein.models.infer import sensor_plus_grid_graph


def _coordinates(points: pd.DataFrame) -> np.ndarray:
    values = points[["lat", "lon"]].to_numpy(float)
    if (not np.isfinite(values).all() or (np.abs(values[:, 0]) > 90).any()
            or (np.abs(values[:, 1]) > 180).any()):
        raise ValueError("registry coordinates must be finite valid latitude and longitude")
    return values


def _ids(values) -> bool:
    return all(isinstance(value, str) and bool(value.strip()) for value in values)


def sensor_support(points: pd.DataFrame, node_ids: list[str], *, k: int,
                   max_km: float) -> tuple[pd.DataFrame, dict]:
    """Measure the actual map graph's structural support at registry grid nodes.

    Only available trained nodes can supply edges. This says nothing about the
    availability or quality of their readings for a particular forecast day.
    """
    if not {"point_id", "kind", "lat", "lon"}.issubset(points):
        raise ValueError("point registry requires point_id, kind, lat and lon")
    if not _ids(points.point_id) or points.point_id.duplicated().any():
        raise ValueError("registry point IDs must be distinct nonempty strings")
    if not isinstance(node_ids, list) or not node_ids or not _ids(node_ids) or len(set(node_ids)) != len(node_ids):
        raise ValueError("model node_ids must be distinct nonempty strings")
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    if isinstance(max_km, bool) or not np.isfinite(max_km) or max_km <= 0:
        raise ValueError("max_km must be finite and positive")
    _coordinates(points)
    registry = points.set_index("point_id")
    available = [node for node in node_ids if node in registry.index]
    if available and registry.loc[available, "kind"].eq("grid").any():
        raise ValueError("trained sensor IDs cannot refer to registry grid rows")
    sensors = registry.loc[available].reset_index()
    grid = points.loc[points.kind.eq("grid"), ["point_id", "lat", "lon"]].reset_index(drop=True)
    if grid.empty:
        raise ValueError("point registry contains no grid nodes")
    sensor_xy, grid_xy = _coordinates(sensors), _coordinates(grid)
    graph = sensor_plus_grid_graph(sensor_xy, grid_xy, k=k, max_km=max_km)
    is_grid = graph.dst >= len(sensors)
    destinations = graph.dst[is_grid] - len(sensors)
    distances = graph.dist_km[is_grid]
    grid["n_sensor_neighbors"] = np.bincount(destinations, minlength=len(grid))
    closest = np.full(len(grid), np.inf)
    farthest = np.full(len(grid), -np.inf)
    np.minimum.at(closest, destinations, distances)
    np.maximum.at(farthest, destinations, distances)
    grid["closest_used_sensor_km"] = np.where(np.isfinite(closest), closest, np.nan)
    grid["farthest_used_sensor_km"] = np.where(np.isfinite(farthest), farthest, np.nan)
    grid["nearest_trained_sensor_km"] = np.nan
    if len(sensors):
        def xyz(xy):
            lat, lon = np.radians(xy).T
            return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

        _, nearest = cKDTree(xyz(sensor_xy)).query(xyz(grid_xy), k=1)
        grid["nearest_trained_sensor_km"] = haversine_km(
            grid_xy[:, 0], grid_xy[:, 1], sensor_xy[nearest, 0], sensor_xy[nearest, 1])
    grid["sensor_support"] = np.where(grid.n_sensor_neighbors > 0, "connected", "no_sensor_edges")
    metadata = {"n_trained_sensors": len(node_ids), "n_available_trained_sensors": len(sensors),
                "missing_trained_sensor_ids": [node for node in node_ids if node not in registry.index],
                "k_neighbors": k, "max_km": float(max_km), "n_grid": len(grid),
                "n_grid_without_sensor_edges": int(grid.n_sensor_neighbors.eq(0).sum()),
                "interpretation": "structural graph support only; current readings and skill are not assessed"}
    return grid, metadata


def audit_forecasts(forecasts: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    """Attach graph support while retaining each saved forecast group's values."""
    required = {"point_id", "kind", "issue_date", "lead", "q50"}
    if not required.issubset(forecasts):
        raise ValueError(f"missing forecast columns: {', '.join(sorted(required - set(forecasts)))}")
    grid = forecasts.loc[forecasts.kind.eq("grid")].copy()
    if grid.empty:
        raise ValueError("no grid forecast rows")
    if (not _ids(grid.point_id) or grid[["issue_date", "lead"]].isna().any().any()
            or grid.duplicated(["point_id", "issue_date", "lead"]).any()):
        raise ValueError("grid forecasts require distinct nonmissing point/date/lead keys")
    if not grid.point_id.isin(support.point_id).all():
        raise ValueError("grid forecast IDs are absent from the audited registry")
    if {"lat", "lon"}.intersection(grid):
        if not {"lat", "lon"}.issubset(grid):
            raise ValueError("forecast coordinates require both lat and lon")
        expected = support.set_index("point_id").loc[grid.point_id, ["lat", "lon"]].to_numpy()
        if not np.allclose(_coordinates(grid), expected, rtol=0, atol=1e-6):
            raise ValueError("forecast coordinates do not match the audited registry")
    collisions = set(support.columns).intersection(grid) - {"point_id", "lat", "lon"}
    if collisions:
        raise ValueError(f"forecast already contains support columns: {', '.join(sorted(collisions))}")
    result = grid.merge(support.drop(columns=["lat", "lon"]) if "lat" in grid else support,
                         on="point_id", how="left", validate="many_to_one")
    values = pd.to_numeric(result.q50, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    result["valid_median"] = np.isfinite(values) & (values >= 0)
    return result


def reading_support(audit: pd.DataFrame, points: pd.DataFrame, node_ids: list[str],
                     features: pd.DataFrame, *, k: int, max_km: float) -> pd.DataFrame:
    """Count neighbor records and finite readings in a supplied as-of feature table."""
    required = {"point_id", "issue_date", "obs_last24", "obs_prevday"}
    if not required.issubset(features):
        raise ValueError(f"missing feature columns: {', '.join(sorted(required - set(features)))}")
    if not _ids(features.point_id):
        raise ValueError("feature point IDs must be nonempty strings")
    features = features.copy()
    result = audit.copy()
    for frame in (features, result):
        try:
            dates = pd.to_datetime(frame.issue_date, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("feature and forecast issue dates must be calendar dates") from exc
        if (not pd.api.types.is_datetime64_any_dtype(dates.dtype)
                or isinstance(dates.dtype, pd.DatetimeTZDtype) or dates.isna().any()
                or not dates.eq(dates.dt.normalize()).all()):
            raise ValueError("feature and forecast issue dates must be calendar dates")
        frame["issue_date"] = dates
    if features.duplicated(["issue_date", "point_id"]).any():
        raise ValueError("duplicate feature point/date records")
    if result.duplicated(["issue_date", "point_id", "lead"]).any():
        raise ValueError("duplicate forecast point/date/lead records")
    for column in ("obs_last24", "obs_prevday"):
        features[column] = pd.to_numeric(features[column], errors="raise").to_numpy(
            dtype=float, na_value=np.nan)
    support, _ = sensor_support(points, node_ids, k=k, max_km=max_km)
    expected = result.point_id.map(support.set_index("point_id").n_sensor_neighbors)
    if expected.isna().any() or not np.array_equal(expected.to_numpy(), result.n_sensor_neighbors.to_numpy()):
        raise ValueError("forecast support must match the supplied registry and graph settings")
    known = set(points.point_id)
    available = [node for node in node_ids if node in known]
    sensors = points.set_index("point_id").loc[available]
    graph = sensor_plus_grid_graph(_coordinates(sensors), _coordinates(support), k=k, max_km=max_km)
    edges = graph.dst >= len(sensors)
    src, dst = graph.src[edges], graph.dst[edges] - len(sensors)
    columns = ["n_neighbors_with_feature_row", "n_neighbors_with_last24", "n_neighbors_with_prevday"]
    if set(columns + ["reading_support"]).intersection(result):
        raise ValueError("forecast audit already contains reading support columns")
    for column in columns:
        result[column] = 0
    for day, indexes in result.groupby("issue_date", sort=True).groups.items():
        daily = features.loc[features.issue_date.eq(day)].set_index("point_id")
        records = np.asarray(sensors.index.isin(daily.index))
        daily = daily.reindex(sensors.index)
        readings = [records, np.isfinite(daily.obs_last24.to_numpy()),
                     np.isfinite(daily.obs_prevday.to_numpy())]
        for column, present in zip(columns, readings):
            counts = pd.Series(np.bincount(dst[present[src]], minlength=len(support)),
                               index=support.point_id)
            result.loc[indexes, column] = result.loc[indexes, "point_id"].map(counts).to_numpy()
    result["reading_support"] = np.select(
        [result.n_sensor_neighbors.eq(0), result.n_neighbors_with_feature_row.eq(0),
         result.n_neighbors_with_last24.eq(0)],
        ["no_sensor_edges", "no_feature_records", "no_last24_readings"], default="last24_available")
    return result


def plot_support(support: pd.DataFrame, sensors: pd.DataFrame, output: Path) -> None:
    """Render support without downloading a basemap or fetching browser assets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, column, title in zip(axes, ["n_sensor_neighbors", "nearest_trained_sensor_km"],
                                   ["Sensor neighbors used", "Nearest trained sensor (km)"]):
        finite = np.isfinite(support[column])
        if finite.any():
            image = axis.scatter(support.loc[finite, "lon"], support.loc[finite, "lat"],
                                  c=support.loc[finite, column], s=8, cmap="viridis")
            fig.colorbar(image, ax=axis, shrink=.8)
        unsupported = support.n_sensor_neighbors.eq(0)
        if unsupported.any():
            axis.scatter(support.loc[unsupported, "lon"], support.loc[unsupported, "lat"],
                          marker="x", c="red", s=12, label="No sensor edges")
        if len(sensors):
            axis.scatter(sensors.lon, sensors.lat, facecolors="none", edgecolors="black",
                          s=20, label="Available trained sensors")
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        mean_lat = float(support.lat.mean())
        axis.set_aspect(1 / max(float(np.cos(np.radians(mean_lat))), .01))
        if unsupported.any() or len(sensors):
            axis.legend(fontsize=7)
    fig.suptitle("Map sensor graph: connectivity and distance")
    try:
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=140)
    finally:
        plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="optional prepared as-of feature table parquet")
    parser.add_argument("--output", type=Path, required=True, help="new report directory")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        study = load_study(args.config)
        model = json.loads(args.model_json.read_text(encoding="utf-8"))
        if not isinstance(model, dict):
            raise TypeError("model metadata must be an object")
        points = pd.read_parquet(args.points)
        support, metadata = sensor_support(points, model.get("node_ids"),
                                            k=study.raw["graph"]["k_neighbors"],
                                            max_km=study.raw["graph"]["max_km"])
        forecasts = audit_forecasts(pd.read_parquet(args.forecasts), support)
        if args.features:
            forecasts = reading_support(forecasts, points, model["node_ids"],
                                         pd.read_parquet(args.features),
                                         k=study.raw["graph"]["k_neighbors"],
                                         max_km=study.raw["graph"]["max_km"])
        counts = forecasts.assign(without_sensor_edges=forecasts.n_sensor_neighbors.eq(0),
                                   invalid_median=~forecasts.valid_median).groupby(
                                       ["issue_date", "lead"], sort=True).agg(
                                           n_grid=("point_id", "size"),
                                           n_without_sensor_edges=("without_sensor_edges", "sum"),
                                           n_invalid_medians=("invalid_median", "sum"))
        if args.features:
            readings = forecasts.assign(
                without_feature_records=forecasts.n_neighbors_with_feature_row.eq(0),
                without_last24_readings=forecasts.n_neighbors_with_last24.eq(0),
                without_prevday_readings=forecasts.n_neighbors_with_prevday.eq(0)).groupby(
                    ["issue_date", "lead"], sort=True)[
                        ["without_feature_records", "without_last24_readings", "without_prevday_readings"]].sum()
            counts = counts.join(readings.add_prefix("n_"))
        args.output.mkdir(parents=True, exist_ok=False)
        support.to_parquet(args.output / "grid_support.parquet", index=False)
        forecasts.to_parquet(args.output / "forecast_audit.parquet", index=False)
        counts.to_csv(args.output / "forecast_counts.csv")
        metadata["sources"] = {key: str(getattr(args, key).resolve()) for key in
                               ["config", "model_json", "points", "forecasts"]}
        if args.features:
            metadata["sources"]["features"] = str(args.features.resolve())
            metadata["reading_check"] = "finite readings in supplied feature rows; as-of provenance and QC not verified"
            metadata["interpretation"] = "structural graph and supplied reading availability; QC and skill not assessed"
        metadata["source_sha256"] = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                      for key, path in metadata["sources"].items()}
        metadata["model_uses_graph"] = model.get("use_graph", True)
        (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        sensors = points.loc[points.point_id.isin(model["node_ids"]) & ~points.kind.eq("grid")]
        plot_support(support, sensors, args.output / "support.png")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Wrote map support audit to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
