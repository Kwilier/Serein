"""Offline map grids and area-weighted summaries of saved grid forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box, covers

from serein.data.points import nearest_site_offset

AREA_CRS = "EPSG:5070"
GROUPS = ["issue_date", "lead", "valid_date"]


def _identifiers(frame: pd.DataFrame, column: str, *, unique: bool = True) -> None:
    if column not in frame:
        raise ValueError(f"missing identifier column: {column}")
    values = frame[column]
    if (values.isna().any() or not values.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
            or (unique and values.duplicated().any())):
        raise ValueError(f"{column} must contain nonempty string identifiers"
                         + (" without duplicates" if unique else ""))


def _polygons(frame: gpd.GeoDataFrame, identifier: str) -> gpd.GeoDataFrame:
    if not isinstance(frame, gpd.GeoDataFrame) or frame.crs is None:
        raise ValueError("polygon tables must be GeoDataFrames with a CRS")
    _identifiers(frame, identifier)
    if (frame.empty or frame.geometry.isna().any() or frame.geometry.is_empty.any()
            or not frame.geom_type.isin(["Polygon", "MultiPolygon"]).all()
            or not frame.geometry.is_valid.all()):
        raise ValueError("polygon tables must contain nonempty valid polygons")
    projected = frame[[identifier, frame.geometry.name]].to_crs(AREA_CRS)
    areas = projected.area.to_numpy()
    if not np.isfinite(projected.total_bounds).all() or not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("polygon areas must be finite and positive in the area CRS")
    return projected.reset_index(drop=True)


def metro_grid(bounds, spacing_km: float = 1., *, max_cells: int = 100_000,
               sites: pd.DataFrame | None = None) -> gpd.GeoDataFrame:
    """Full Albers squares intersecting a CONUS lon/lat box, with forecast centers.

    Cell edges are anchored at the projection origin so overlapping requests use
    identical cells and point IDs. Edge cells extend outside the requested box.
    """
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.shape != (4,) or not np.isfinite(bounds).all()
            or not (-125 <= bounds[0] < bounds[2] <= -66)
            or not (24 <= bounds[1] < bounds[3] <= 50)):
        raise ValueError("bounds must be west south east north within CONUS (-125, 24, -66, 50)")
    if isinstance(spacing_km, bool) or not np.isfinite(spacing_km) or spacing_km <= 0:
        raise ValueError("spacing_km must be finite and positive")
    if isinstance(max_cells, bool) or not isinstance(max_cells, int) or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    # Millimeter resolution keeps IDs exact and stable across equivalent inputs.
    step_mm = round(spacing_km * 1_000_000)
    if step_mm < 1 or not np.isclose(spacing_km * 1_000_000, step_mm, rtol=0, atol=1e-6):
        raise ValueError("spacing_km must be a whole number of millimeters")
    step = step_mm / 1000
    extent = gpd.GeoSeries([box(*bounds)], crs=4326).segmentize(.01).to_crs(AREA_CRS).iloc[0]
    west, south, east, north = extent.bounds
    x0, y0 = np.floor(np.array([west, south]) / step).astype(np.int64)
    x1, y1 = np.ceil(np.array([east, north]) / step).astype(np.int64)
    count = int(x1 - x0) * int(y1 - y0)
    if count > max_cells:
        raise ValueError(f"grid requires up to {count} cells, exceeding max_cells={max_cells}")
    ix, iy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    ix, iy = ix.ravel(), iy.ravel()
    geometry = gpd.GeoSeries(box(ix * step, iy * step, (ix + 1) * step, (iy + 1) * step),
                             crs=AREA_CRS)
    keep = geometry.intersection(extent).area.to_numpy() > 0
    ix, iy, geometry = ix[keep], iy[keep], geometry[keep].reset_index(drop=True)
    centers = geometry.centroid.to_crs(4326)
    if sites is not None and "kind" in sites:
        sites = sites.loc[sites.kind.eq("epa")]
    return gpd.GeoDataFrame({
        "point_id": [f"GA5070_{step_mm}_{x}_{y}" for x, y in zip(ix, iy)],
        "kind": "grid", "lat": centers.y, "lon": centers.x,
        "gmt_offset": nearest_site_offset(centers.y, centers.x, sites),
        "spacing_km": spacing_km,
    }, geometry=geometry, crs=AREA_CRS)


def grid_registry(cells: gpd.GeoDataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Replace grid rows in a registry while preserving the original sensor rows."""
    _identifiers(cells, "point_id")
    _identifiers(points, "point_id")
    required = {"lat", "lon", "gmt_offset", "kind"}
    if not required.issubset(cells) or not required.issubset(points):
        raise ValueError("registries require lat, lon, gmt_offset and kind columns")
    if not cells.kind.eq("grid").all():
        raise ValueError("new cells must all have kind=grid")
    sensors = points.loc[~points.kind.eq("grid")].copy()
    new = pd.DataFrame(cells.drop(columns=cells.geometry.name))
    result = pd.concat([sensors, new], ignore_index=True)
    _identifiers(result, "point_id")
    return result


def degree_grid_cells(points: pd.DataFrame, lat_step: float, lon_step: float) -> gpd.GeoDataFrame:
    """Explicit footprints for an existing regular lon/lat map grid.

    Supply the original configured spacing, including when rows are missing.
    Inferring spacing from surviving points would expand holes into large cells.
    """
    for step in (lat_step, lon_step):
        if isinstance(step, bool) or not np.isfinite(step) or step <= 0:
            raise ValueError("grid spacing must be finite and positive")
    required = {"point_id", "lat", "lon"}
    if not required.issubset(points):
        raise ValueError("grid points require point_id, lat and lon")
    points = points.loc[points["kind"].eq("grid")] if "kind" in points else points
    _identifiers(points, "point_id")
    xy = points[["lon", "lat"]].to_numpy(float)
    if (not len(xy) or not np.isfinite(xy).all()
            or (np.abs(xy[:, 0]) + lon_step / 2 > 180).any()
            or (np.abs(xy[:, 1]) + lat_step / 2 >= 90).any()):
        raise ValueError("grid cell coordinates and edges must be finite and within lon/lat bounds")
    if points.duplicated(["lat", "lon"]).any():
        raise ValueError("grid centers must be distinct")
    geometry = box(xy[:, 0] - lon_step / 2, xy[:, 1] - lat_step / 2,
                   xy[:, 0] + lon_step / 2, xy[:, 1] + lat_step / 2)
    return gpd.GeoDataFrame(points[["point_id"]].reset_index(drop=True),
                            geometry=geometry, crs=4326)


def tract_weights(cells: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame,
                  tract_id: str = "GEOID") -> pd.DataFrame:
    """Positive intersection areas in square meters, without double-counting cells."""
    cells = _polygons(cells, "point_id")
    tracts = _polygons(tracts, tract_id)
    pairs = cells.sindex.query(cells.geometry, predicate="intersects")
    first, second = pairs[:, pairs[0] < pairs[1]]
    if len(first):
        overlap = cells.geometry.iloc[first].reset_index(drop=True).intersection(
            cells.geometry.iloc[second].reset_index(drop=True)).area.to_numpy()
        tolerance = np.minimum(cells.area.to_numpy()[first], cells.area.to_numpy()[second]) * 1e-9
        if (overlap > tolerance).any():
            raise ValueError("grid cells overlap in area")
    ti, ci = cells.sindex.query(tracts.geometry, predicate="intersects")
    area = tracts.geometry.iloc[ti].reset_index(drop=True).intersection(
        cells.geometry.iloc[ci].reset_index(drop=True)).area.to_numpy()
    keep = area > 0
    return pd.DataFrame({"tract_id": tracts[tract_id].to_numpy()[ti[keep]],
                         "point_id": cells.point_id.to_numpy()[ci[keep]],
                         "area_m2": area[keep]})


def _forecast_rows(forecasts: pd.DataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
    required = {"point_id", "q50", *GROUPS}
    if not required.issubset(forecasts):
        raise ValueError(f"missing forecast columns: {', '.join(sorted(required - set(forecasts)))}")
    frame = forecasts.loc[forecasts["kind"].eq("grid")].copy() if "kind" in forecasts else forecasts.copy()
    if frame.empty:
        raise ValueError("no grid forecast rows")
    _identifiers(frame, "point_id", unique=False)
    if not frame.point_id.isin(cells.point_id).all():
        raise ValueError("forecast point IDs are absent from grid cells")
    coordinates = {"lat", "lon"}.intersection(frame)
    if coordinates:
        if coordinates != {"lat", "lon"}:
            raise ValueError("forecast coordinates require both lat and lon")
        checked = frame[["point_id", "lon", "lat"]].drop_duplicates()
        xy = checked[["lon", "lat"]].to_numpy(float)
        if (not np.isfinite(xy).all() or (np.abs(xy[:, 0]) > 180).any()
                or (np.abs(xy[:, 1]) >= 90).any()):
            raise ValueError("forecast coordinates must be finite valid lon/lat")
        locations = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=4326).to_crs(AREA_CRS)
        footprints = cells.set_index("point_id").geometry.reindex(checked.point_id)
        inside = covers(footprints.to_numpy(), locations.to_numpy())
        if not inside.all():
            outside = np.flatnonzero(~inside)
            if not covers(footprints.iloc[outside].buffer(.001).to_numpy(),
                          locations.iloc[outside].to_numpy()).all():
                raise ValueError("forecast coordinates fall outside their grid cell footprints")
    for column in ("issue_date", "valid_date"):
        try:
            dates = pd.to_datetime(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} must contain timezone-free calendar dates") from exc
        if (not pd.api.types.is_datetime64_any_dtype(dates.dtype)
                or isinstance(dates.dtype, pd.DatetimeTZDtype) or dates.isna().any()
                or not dates.eq(dates.dt.normalize()).all()):
            raise ValueError(f"{column} must contain timezone-free calendar dates")
        frame[column] = dates
    leads = pd.to_numeric(frame.lead, errors="raise")
    if (frame.lead.map(lambda value: isinstance(value, (bool, np.bool_))).any()
            or not np.isfinite(leads).all()
            or (leads < 0).any() or (leads % 1 != 0).any()):
        raise ValueError("lead must contain nonnegative integers")
    frame["lead"] = leads.astype(int)
    if not (frame.valid_date - frame.issue_date).eq(pd.to_timedelta(frame.lead, unit="D")).all():
        raise ValueError("valid_date must equal issue_date plus lead days")
    if frame.duplicated([*GROUPS, "point_id"]).any():
        raise ValueError("duplicate grid forecasts for the same issue date and lead")
    values = pd.to_numeric(frame.q50, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("q50 must be nonnegative and finite, or missing")
    frame["q50"] = values
    return frame


def aggregate_tracts(forecasts: pd.DataFrame, cells: gpd.GeoDataFrame,
                     tracts: gpd.GeoDataFrame, *, tract_id: str = "GEOID",
                     min_coverage: float = .95) -> gpd.GeoDataFrame:
    """Area-weight pointwise medians, retaining missing and out-of-grid tracts.

    The result is a spatial average of medians, not a quantile of a tract-wide
    random variable. No aggregated uncertainty interval is inferred.
    """
    if isinstance(min_coverage, bool) or not np.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be finite and in (0, 1]")
    projected = _polygons(tracts, tract_id)
    cells = _polygons(cells, "point_id")
    weights = tract_weights(cells, projected, tract_id)
    frame = _forecast_rows(forecasts, cells)
    indexed = projected.set_index(tract_id)
    tract_area = indexed.area
    grid_area = weights.groupby("tract_id").area_m2.sum().reindex(indexed.index, fill_value=0)
    records = []
    for group, forecast in frame.groupby(GROUPS, sort=True):
        available = weights.merge(forecast[["point_id", "q50"]], on="point_id", how="left",
                                  validate="many_to_one").dropna(subset=["q50"])
        valid_area = available.groupby("tract_id").area_m2.sum().reindex(indexed.index, fill_value=0)
        coverage = (valid_area / tract_area).clip(0, 1)
        # Normalize before multiplying to avoid overflow from concentration * area.
        denominators = valid_area.reindex(available.tract_id).to_numpy(float)
        available["weighted"] = (available.area_m2.to_numpy() / denominators) * available.q50.to_numpy()
        means = available.groupby("tract_id").weighted.sum().reindex(indexed.index)
        accepted = (valid_area > 0) & (coverage >= min_coverage - 1e-12)
        status = np.where(valid_area == 0, "no_forecast",
                          np.where(accepted, "ok", "insufficient_coverage"))
        records.append(gpd.GeoDataFrame({
            "tract_id": indexed.index, **dict(zip(GROUPS, group)),
            "q50_area_mean": means.where(accepted).to_numpy(),
            "tract_area_m2": tract_area.to_numpy(),
            "grid_area_m2": grid_area.to_numpy(), "available_area_m2": valid_area.to_numpy(),
            "grid_coverage": (grid_area / tract_area).clip(0, 1).to_numpy(),
            "forecast_coverage": coverage.to_numpy(),
            "n_cells": weights.groupby("tract_id").size().reindex(indexed.index, fill_value=0).to_numpy(),
            "n_available": available.groupby("tract_id").size().reindex(indexed.index, fill_value=0).to_numpy(),
            "min_coverage": min_coverage, "status": status,
        }, geometry=tracts.geometry.to_numpy(), crs=tracts.crs))
    return gpd.GeoDataFrame(pd.concat(records, ignore_index=True), crs=tracts.crs)


def plot_cell_forecasts(forecasts: pd.DataFrame, cells: gpd.GeoDataFrame, output: Path,
                        *, max_panels: int = 12) -> None:
    """Draw actual footprints instead of treating projected centers as a degree raster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    cells = _polygons(cells, "point_id")
    forecasts = _forecast_rows(forecasts, cells)
    groups = list(forecasts.groupby(GROUPS, sort=True))
    if isinstance(max_panels, bool) or not isinstance(max_panels, int) or max_panels < 1:
        raise ValueError("max_panels must be a positive integer")
    if len(groups) > max_panels:
        raise ValueError(f"forecast has {len(groups)} panels, exceeding max_panels={max_panels}")
    ncols = min(4, len(groups))
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False, layout="constrained")
    values = forecasts.q50.dropna()
    norm = Normalize(vmin=0, vmax=max(float(values.max()) if len(values) else 0, 1.))
    geographic = cells.to_crs(4326).set_index("point_id")
    try:
        for axis, (group, forecast) in zip(axes.ravel(), groups):
            view = geographic.join(forecast.set_index("point_id")[["q50"]], how="left")
            view.plot(column="q50", ax=axis, cmap="viridis", norm=norm,
                      edgecolor="none", missing_kwds={"color": "lightgray", "hatch": "///"})
            if view.q50.isna().any():
                axis.legend(handles=[Patch(facecolor="lightgray", hatch="///", label="No forecast")],
                            fontsize=7)
            issue, lead, valid = group
            axis.set_title(f"Issued {issue:%Y-%m-%d}, day +{lead}\nValid {valid:%Y-%m-%d}")
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
        for axis in axes.ravel()[len(groups):]:
            axis.set_visible(False)
        fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes.ravel()[:len(groups)]),
                     shrink=.75, label="Cell median PM2.5 (ug/m3)")
        fig.suptitle("Saved grid forecasts: cell median PM2.5")
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=140)
    finally:
        plt.close(fig)


def forecast_neighborhood(config: Path, day: date, run: Path, cell_path: Path, output: Path) -> None:
    """Export a prepared development grid with its footprints and support report."""
    from serein.config import load_study
    from serein.data.points import load_points
    from serein.eval.map_support import sensor_support
    from serein.models.infer import forecast_day

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    study = load_study(config)
    if (day.year not in study.raw["split"]["calibrate"]
            or day.month not in study.raw["season_months"]):
        raise ValueError("forecast date must fall in a configured calibration season")
    cells = gpd.read_parquet(cell_path)
    projected = _polygons(cells, "point_id")
    points = load_points(study.name)
    _identifiers(points, "point_id")
    grid = points.loc[points.kind.eq("grid")]
    if set(grid.point_id) != set(cells.point_id):
        raise ValueError("study registry grid IDs must match the supplied cells exactly")
    if not study.in_study(grid.lat, grid.lon).all():
        raise ValueError("grid centers must obey the study region and exclusion rules")
    # Missing NOAA rows would otherwise become imputed model inputs and could
    # produce finite predictions on a grid that has never been extracted.
    from serein.data import aqm, hrrr

    model_meta = json.loads((run / "model.json").read_text(encoding="utf-8"))
    if not isinstance(model_meta, dict):
        raise TypeError("model metadata must be an object")
    history = model_meta.get("history", 1)
    if isinstance(history, bool) or not isinstance(history, int) or history < 1:
        raise ValueError("model history must be a positive integer")
    point_hash = hashlib.sha256(pd.util.hash_pandas_object(
        points[["point_id", "lat", "lon"]], index=False).to_numpy().tobytes()).hexdigest()
    for name, loader, cycle in (("HRRR", hrrr.load_cycles, study.raw["availability"]["hrrr_cycle"]),
                                ("AQM", aqm.load_cycles, study.raw["availability"]["aqm_cycle"])):
        for back in range(history):
            input_day = day - timedelta(days=back)
            cached = loader([input_day], cycle=cycle, store=study.name)
            if cached.empty or not set(grid.point_id).issubset(cached.point_id):
                raise ValueError(f"{name} extraction must include every grid point for {input_day}")
            cached_hash = (cached.attrs.get("hrrr_request", {}).get("points") if name == "HRRR"
                           else cached.attrs.get("aqm_points"))
            if cached_hash != point_hash:
                raise ValueError(f"{name} extraction metadata must match the current point registry")
    support, metadata = sensor_support(points, model_meta.get("node_ids"),
                                        k=study.raw["graph"]["k_neighbors"],
                                        max_km=study.raw["graph"]["max_km"])
    forecast = forecast_day(study, day, run)
    _forecast_rows(forecast, projected)
    output.mkdir(parents=True, exist_ok=False)
    forecast.to_parquet(output / "forecast.parquet", index=False)
    cells.to_parquet(output / "cells.parquet", index=False)
    support.to_parquet(output / "grid_support.parquet", index=False)
    plot_cell_forecasts(forecast, cells, output / "map.png")
    metadata.update(issue_date=day.isoformat(), model_members=model_meta.get("n_members", 1),
                    history_days=history,
                    model_seeds=model_meta.get("seeds"), registry_points_sha256=point_hash,
                    sources={"config": str(config.resolve()), "model_run": str(run.resolve()),
                             "cells": str(cell_path.resolve())},
                    input_check="grid IDs and point hashes checked for each history day; completeness and skill not assessed")
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    grid = sub.add_parser("grid", help="make projected kilometer grid cells")
    grid.add_argument("--bounds", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    grid.add_argument("--spacing-km", type=float, default=1.)
    grid.add_argument("--max-cells", type=int, default=100_000)
    grid.add_argument("--sites", type=Path, help="point registry for nearest-monitor time zones")
    grid.add_argument("--output", type=Path, required=True, help="new GeoParquet file")
    grid.add_argument("--registry-output", type=Path, help="new point registry; requires --sites")
    aggregate = sub.add_parser("tracts", help="summarize saved grid forecasts over local polygons")
    aggregate.add_argument("--forecasts", type=Path, required=True)
    source = aggregate.add_mutually_exclusive_group(required=True)
    source.add_argument("--cells", type=Path, help="GeoParquet cells with point_id")
    source.add_argument("--points", type=Path, help="regular degree-grid registry parquet")
    aggregate.add_argument("--lat-step", type=float)
    aggregate.add_argument("--lon-step", type=float)
    aggregate.add_argument("--tracts", type=Path, required=True, help="local polygon file or GeoParquet")
    aggregate.add_argument("--tract-id", default="GEOID")
    aggregate.add_argument("--min-coverage", type=float, default=.95)
    aggregate.add_argument("--output", type=Path, required=True, help="new GeoParquet file")
    plot = sub.add_parser("plot", help="draw projected grid footprints without a regular-degree raster")
    plot.add_argument("--forecasts", type=Path, required=True)
    plot.add_argument("--cells", type=Path, required=True, help="GeoParquet cells with point_id")
    plot.add_argument("--max-panels", type=int, default=12)
    plot.add_argument("--output", type=Path, required=True, help="new PNG file")
    forecast = sub.add_parser("forecast", help="export a prepared grid for a development issue date")
    forecast.add_argument("--config", type=Path, required=True)
    forecast.add_argument("--date", type=date.fromisoformat, required=True)
    forecast.add_argument("--run", type=Path, required=True, help="saved model directory")
    forecast.add_argument("--cells", type=Path, required=True)
    forecast.add_argument("--output", type=Path, required=True, help="new export directory")
    args = parser.parse_args(argv)
    try:
        destinations = [args.output]
        if args.action == "grid" and args.registry_output is not None:
            if args.sites is None:
                raise ValueError("--registry-output requires --sites")
            destinations.append(args.registry_output)
        if len({path.resolve() for path in destinations}) != len(destinations):
            raise ValueError("grid and registry output paths must differ")
        for path in destinations:
            if path.exists():
                raise ValueError(f"output already exists: {path}")
            if not path.parent.is_dir():
                raise ValueError(f"output parent does not exist: {path.parent}")
        if args.action == "grid":
            sites = pd.read_parquet(args.sites) if args.sites else None
            result = metro_grid(args.bounds, args.spacing_km, max_cells=args.max_cells, sites=sites)
            registry = grid_registry(result, sites) if args.registry_output else None
        elif args.action == "plot":
            plot_cell_forecasts(pd.read_parquet(args.forecasts), gpd.read_parquet(args.cells),
                                 args.output, max_panels=args.max_panels)
            print(f"Wrote cell forecast map to {args.output}")
            return 0
        elif args.action == "forecast":
            forecast_neighborhood(args.config, args.date, args.run, args.cells, args.output)
            print(f"Wrote neighborhood forecast to {args.output}")
            return 0
        else:
            if args.points:
                if args.lat_step is None or args.lon_step is None:
                    raise ValueError("--points requires --lat-step and --lon-step")
                cells = degree_grid_cells(pd.read_parquet(args.points), args.lat_step, args.lon_step)
            else:
                if args.lat_step is not None or args.lon_step is not None:
                    raise ValueError("grid steps only apply with --points")
                cells = gpd.read_parquet(args.cells)
            tracts = (gpd.read_parquet(args.tracts) if args.tracts.suffix.lower() == ".parquet"
                      else gpd.read_file(args.tracts))
            result = aggregate_tracts(pd.read_parquet(args.forecasts), cells, tracts,
                                      tract_id=args.tract_id, min_coverage=args.min_coverage)
        with args.output.open("xb") as stream:
            result.to_parquet(stream, index=False)
        if args.action == "grid" and args.registry_output:
            with args.registry_output.open("xb") as stream:
                registry.to_parquet(stream, index=False)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(result)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
