"""Projected registries use the real ensemble and keep grids out of sensor messages."""

from datetime import date, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from serein.config import Study
from serein.data.points import save_points
from serein.features.prepare import Prep
from serein.models import infer
from serein.models.conformal import CQR
from serein.models.gnn import SmokeCastGNN
from serein.models.train import TrainConfig
from serein.spatial import aggregate_tracts, grid_registry, metro_grid


def _table(days, points):
    parts = []
    for day in days:
        frame = points[["point_id", "kind", "lat", "lon"]].copy()
        median = 20 + (frame.lon + 122) * 5
        sensor = frame.kind.ne("grid")
        frame["issue_date"] = day
        frame["hms_prev"] = 0.
        frame["obs_last24"] = median.where(sensor)
        frame["obs_prevday"] = (median - 1).where(sensor)
        frame["hrrr_u10_L0"], frame["hrrr_v10_L0"] = 2., 1.
        for lead in (0, 1):
            frame[f"aqm_mean_L{lead}"] = median + lead + 3
            frame[f"y_L{lead}"] = (median + lead).where(sensor)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_projected_grid_ensemble_forecasts_are_invariant_to_other_grid_nodes(tmp_path, monkeypatch):
    monkeypatch.setenv("SEREIN_DATA", str(tmp_path / "data"))
    sensors = pd.DataFrame({"point_id": ["A", "B", "C"], "kind": "epa", "gmt_offset": -8,
                            "lat": [38.48, 38.53, 38.58], "lon": [-121.58, -121.53, -121.48]})
    small = metro_grid([-121.55, 38.5, -121.5, 38.55], 1, sites=sensors)
    large = metro_grid([-121.6, 38.45, -121.45, 38.6], 1, sites=sensors)
    for name, cells in (("small", small), ("large", large)):
        save_points(grid_registry(cells, sensors), name)
    train = _table([date(2021, 8, 1) + timedelta(days=offset) for offset in range(4)], sensors)
    prep = Prep().fit(train)
    features, _ = prep.transform(train)
    cfg = TrainConfig(hidden=8, layers=2, heads=2, dropout=0., seed=7)
    states = []
    for seed in (7, 8):
        torch.manual_seed(seed)
        model = SmokeCastGNN(features.shape[1], 2, hidden=8, layers=2, heads=2, drop=0.)
        states.append(model.state_dict())
    run = tmp_path / "model"
    infer.save_artifacts(run, prep, sensors.point_id.to_list(), features.shape[1], [0, 1], cfg,
                         CQR(qhat={(0, -1): .1, (1, -1): .1}), states, seeds=[7, 8])
    seen = []

    def build_table(study, days, points, **kwargs):
        frame = _table(days, points)
        assert frame.loc[frame.kind.eq("grid"), ["obs_last24", "obs_prevday", "y_L0", "y_L1"]].isna().all().all()
        seen.append((study.name, len(points)))
        return frame

    monkeypatch.setattr(infer, "build_table", build_table)
    forecasts = []
    for name in ("small", "large"):
        study = Study({"name": name, "leads": [0, 1], "graph": {"k_neighbors": 2, "max_km": 100}})
        forecasts.append(infer.forecast_day(study, date(2023, 8, 18), run))
    assert seen == [("small", len(sensors) + len(small)), ("large", len(sensors) + len(large))]
    columns = ["q05", "q25", "q50", "q75", "q95"]
    small_forecast, large_forecast = [frame.set_index(["point_id", "lead"]) for frame in forecasts]
    np.testing.assert_allclose(small_forecast[columns], large_forecast.loc[small_forecast.index, columns],
                               rtol=1e-5, atol=1e-5)
    quantiles = large_forecast[columns].to_numpy()
    assert np.isfinite(quantiles).all()
    assert (quantiles >= 0).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()
    tract = gpd.GeoDataFrame({"GEOID": ["synthetic"]}, geometry=[small.geometry.union_all()], crs=small.crs)
    result = aggregate_tracts(forecasts[0], small, tract)
    expected = forecasts[0].loc[forecasts[0].kind.eq("grid")].groupby("lead").q50.mean()
    np.testing.assert_allclose(result.q50_area_mean, result.lead.map(expected))
    np.testing.assert_allclose(result.forecast_coverage, 1.)
