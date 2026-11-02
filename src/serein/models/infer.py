"""Use a trained SmokeCast model: forecasts for any day, at sensors AND at map grid points.

Grid points are "virtual sensors": they have NOAA forecasts, weather, fires and
smoke plumes, but no readings. Each grid point receives messages from its k
nearest real sensors (never from other grid points), the same way a sensor
with hidden readings did during training (node masking).
"""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from serein.aqi import pm25_to_category
from serein.config import Study
from serein.data.firms import bearing_rad, haversine_km
from serein.data.points import load_points
from serein.features.graph import Graph, build_knn_graph
from serein.features.prepare import leads_of, neighbor_base, to_graph_data
from serein.features.table import build_table
from serein.models.gnn import QUANTILES, SmokeCastGNN

AQI_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]


def save_artifacts(out_dir: Path, prep, node_ids, n_feat, leads, cfg, cqr, state_dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out_dir / "smokecast.pt")
    with open(out_dir / "artifacts.pkl", "wb") as f:
        pickle.dump({"prep": prep, "cqr": cqr}, f)
    (out_dir / "model.json").write_text(json.dumps({
        "node_ids": list(node_ids), "n_feat": int(n_feat), "leads": list(leads),
        "hidden": cfg.hidden, "layers": cfg.layers, "heads": cfg.heads}, indent=1))


def load_artifacts(run_dir: Path):
    meta = json.loads((run_dir / "model.json").read_text())
    with open(run_dir / "artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    model = SmokeCastGNN(meta["n_feat"], len(meta["leads"]), meta["hidden"], meta["layers"],
                         meta["heads"], drop=0.0)
    model.load_state_dict(torch.load(run_dir / "smokecast.pt", map_location="cpu"))
    model.eval()
    return model, art["prep"], art["cqr"], meta


def sensor_plus_grid_graph(sensor_xy, grid_xy, k: int, max_km: float) -> Graph:
    """Sensors keep their sensor-only kNN graph; each grid point listens to k sensors."""
    gs = build_knn_graph(sensor_xy[:, 0], sensor_xy[:, 1], k=k, max_km=max_km)
    ns = len(sensor_xy)

    def xyz(a):
        la, lo = np.radians(a[:, 0]), np.radians(a[:, 1])
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)

    _, nbr = cKDTree(xyz(sensor_xy)).query(xyz(grid_xy), k=k)
    src = nbr.reshape(-1)
    dst = np.repeat(np.arange(len(grid_xy)), k) + ns
    d = haversine_km(sensor_xy[src, 0], sensor_xy[src, 1], grid_xy[dst - ns, 0], grid_xy[dst - ns, 1])
    keep = d <= max_km
    src, dst, d = src[keep], dst[keep], d[keep]
    b = bearing_rad(sensor_xy[src, 0], sensor_xy[src, 1], grid_xy[dst - ns, 0], grid_xy[dst - ns, 1])
    attr = np.stack([d / max_km, np.log1p(d) / np.log1p(max_km), np.sin(b), np.cos(b)], 1)
    return Graph(np.r_[gs.src, src], np.r_[gs.dst, dst], np.r_[gs.dist_km, d],
                 np.r_[gs.bearing, b], np.r_[gs.edge_attr, attr].astype(np.float32))


def forecast_day(study: Study, day: date, run_dir: Path, with_grid: bool = True) -> pd.DataFrame:
    model, prep, cqr, meta = load_artifacts(run_dir)
    pts = load_points(study.name).set_index("point_id")
    sensors = [n for n in meta["node_ids"] if n in pts.index]
    grid = pts.index[pts["kind"] == "grid"].tolist() if with_grid else []
    use = pts.loc[sensors + grid].reset_index()
    tab = build_table(study, [day], use, progress=False)
    tab = tab.set_index("point_id").reindex(sensors + grid).reset_index()
    tab["issue_date"] = day
    for c in [c for c in tab.columns if c.startswith("y_L")]:
        tab[c] = tab[c].astype(float)
    X, _ = prep.transform(tab)
    leads = leads_of(tab)
    base = prep.base(tab, leads)
    wind = np.nan_to_num(tab[["hrrr_u10_L0", "hrrr_v10_L0"]].to_numpy(np.float32)) \
        if "hrrr_u10_L0" in tab else np.zeros((len(tab), 2), np.float32)
    sxy = pts.loc[sensors, ["lat", "lon"]].to_numpy(float)
    gxy = pts.loc[grid, ["lat", "lon"]].to_numpy(float) if grid else np.zeros((0, 2))
    k, mk = study.raw["graph"]["k_neighbors"], study.raw["graph"]["max_km"]
    g = sensor_plus_grid_graph(sxy, gxy, k, mk) if grid else build_knn_graph(sxy[:, 0], sxy[:, 1], k, mk)
    if grid:
        # Grid points have no readings: their base comes from nearby sensors' readings.
        gd = to_graph_data(tab, prep, sensors + grid)
        is_sensor = np.r_[np.ones(len(sensors), bool), np.zeros(len(grid), bool)]
        b_nbr = neighbor_base(gd, tab, prep, g, sources=is_sensor)[0]
        base = np.where(is_sensor[:, None], base, b_nbr).astype(np.float32)
    t = lambda a: torch.as_tensor(a).unsqueeze(0)
    with torch.no_grad():
        q = model(t(X), t(wind), t(base), torch.as_tensor(g.src), torch.as_tensor(g.dst),
                  torch.as_tensor(g.bearing, dtype=torch.float32), torch.as_tensor(g.edge_attr))[0].numpy()
    lo, hi = cqr.apply(q[..., 0], q[..., -1], q[..., 2])
    q[..., 0], q[..., -1] = np.minimum(lo, q[..., 2]), np.maximum(hi, q[..., 2])
    rows = []
    for li, lead in enumerate(leads):
        valid = day + pd.Timedelta(days=lead).to_pytimedelta()
        med = np.expm1(q[:, li, 2])
        rows.append(pd.DataFrame({
            "point_id": tab["point_id"], "kind": use.set_index("point_id").loc[tab["point_id"], "kind"].to_numpy(),
            "lat": tab["lat"], "lon": tab["lon"], "issue_date": day, "lead": lead, "valid_date": valid,
            **{f"q{int(100 * lv):02d}": np.expm1(q[:, li, j]) for j, lv in enumerate(QUANTILES)},
            "aqi_cat": pm25_to_category(med, valid), "obs": tab.get(f"y_L{lead}"),
        }))
    return pd.concat(rows, ignore_index=True)


STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"


def state_boundaries():
    """US state outlines (Census cartographic boundary file, free), cached."""
    import geopandas as gpd

    from serein.config import raw_dir
    from serein.http import get

    path = raw_dir("boundaries") / "cb_2023_us_state_20m.zip"
    if not path.exists():
        try:
            path.write_bytes(get(STATES_URL).content)
        except Exception:  # noqa: BLE001 - maps still work without outlines
            return None
    return gpd.read_file(f"zip://{path}").to_crs(4326)


def plot_map(fc: pd.DataFrame, out_png: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    leads = sorted(fc["lead"].unique())
    fig, axes = plt.subplots(1, len(leads), figsize=(4.2 * len(leads), 6), squeeze=False)
    cmap = ListedColormap(AQI_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
    states = state_boundaries()
    land = None
    if states is not None:
        import geopandas as gpd
        g0 = fc[(fc["kind"] == "grid") & (fc["lead"] == leads[0])]
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(g0["lon"], g0["lat"]), crs=4326)
        inside = gpd.sjoin(pts, states[["geometry"]], predicate="within", how="inner").index
        land = set(g0["point_id"].iloc[np.unique(inside)])
    for ax, lead in zip(axes[0], leads):
        d = fc[fc["lead"] == lead]
        gr, se = d[d["kind"] == "grid"], d[d["kind"] != "grid"]
        if land is not None:
            gr = gr[gr["point_id"].isin(land)]
        if len(gr):
            # Draw grid cells as an image so they tile without gaps.
            lats, lons = np.sort(gr["lat"].unique()), np.sort(gr["lon"].unique())
            img = np.full((len(lats), len(lons)), np.nan)
            img[np.searchsorted(lats, gr["lat"]), np.searchsorted(lons, gr["lon"])] = gr["aqi_cat"]
            step = lats[1] - lats[0] if len(lats) > 1 else 0.25
            ax.imshow(img, origin="lower", cmap=cmap, norm=norm, alpha=0.75, interpolation="nearest",
                      extent=(lons[0] - step / 2, lons[-1] + step / 2, lats[0] - step / 2, lats[-1] + step / 2))
        if states is not None:
            states.boundary.plot(ax=ax, color="k", linewidth=0.5)
        ax.scatter(se["lon"], se["lat"], c=se["aqi_cat"], cmap=cmap, norm=norm, s=14,
                   edgecolors="k", linewidths=0.4)
        ax.set_xlim(d["lon"].min() - 0.3, d["lon"].max() + 0.3)
        ax.set_ylim(d["lat"].min() - 0.3, d["lat"].max() + 0.3)
        ax.set_title(f"{d['valid_date'].iloc[0]} (day +{lead})")
        ax.set_aspect(1 / np.cos(np.radians(d["lat"].mean())))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(title or "SmokeCast forecast: daily PM2.5 AQI category (squares = map grid, dots = sensors)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
