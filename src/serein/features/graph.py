"""The sensor graph: who listens to whom.

Each point is a node. Each node gets edges FROM its k nearest neighbors (within
``max_km``). Edge features tell the model how far away the neighbor is and in
which direction, so combined with the wind it can learn "smoke at my upwind
neighbor now = smoke here soon".

Example: Sacramento's neighbors include Davis (~20 km W) and Roseville (~25 km NE).
With a NE wind, Roseville is upwind and should matter more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from serein.data.firms import bearing_rad, haversine_km


@dataclass
class Graph:
    src: np.ndarray        # [E] neighbor index (message sender)
    dst: np.ndarray        # [E] receiving node
    dist_km: np.ndarray    # [E]
    bearing: np.ndarray    # [E] radians, direction from src to dst (0 = north)
    edge_attr: np.ndarray  # [E, 4] static edge features

    @property
    def n_edges(self) -> int:
        return len(self.src)


def build_knn_graph(lat, lon, k: int = 10, max_km: float = 300.0) -> Graph:
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    xyz = np.stack([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                    np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                    np.sin(np.radians(lat))], 1)
    tree = cKDTree(xyz)
    kk = min(k + 1, len(lat))
    _, nbr = tree.query(xyz, k=kk)
    src, dst = [], []
    for i in range(len(lat)):
        for j in np.atleast_1d(nbr[i])[1:]:
            src.append(j)
            dst.append(i)
    src, dst = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)
    d = haversine_km(lat[src], lon[src], lat[dst], lon[dst])
    keep = d <= max_km
    src, dst, d = src[keep], dst[keep], d[keep]
    b = bearing_rad(lat[src], lon[src], lat[dst], lon[dst])
    attr = np.stack([d / max_km, np.log1p(d) / np.log1p(max_km), np.sin(b), np.cos(b)], 1)
    return Graph(src, dst, d, b, attr.astype(np.float32))


def idw_neighbor_mean(values: np.ndarray, g: Graph, power: float = 1.0) -> np.ndarray:
    """Inverse-distance-weighted mean of neighbors' values (NaN-aware).

    values: [..., N] -> [..., N]
    """
    w = 1.0 / np.maximum(g.dist_km, 1.0) ** power
    v = values[..., g.src]
    ok = np.isfinite(v)
    num = np.zeros(values.shape, float)
    den = np.zeros(values.shape, float)
    np.add.at(num.T, g.dst, (np.where(ok, v, 0) * w).T)
    np.add.at(den.T, g.dst, (ok * w).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)
