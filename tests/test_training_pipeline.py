"""Small real training/export/inference checks; no external data or final evaluation."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from serein.config import Study
from serein.eval import backtest
from serein.features.graph import build_knn_graph
from serein.features.prepare import to_graph_data
from serein.models import infer


def _synthetic_table(start, n_days, points):
    rows = []
    for offset in range(n_days):
        for site, point in points.iterrows():
            pm = 15 + 4 * np.sin(offset / 2) + site * 3
            rows.append({
                "issue_date": start + timedelta(days=offset), "point_id": point.point_id,
                "kind": "epa", "lat": point.lat, "lon": point.lon,
                "hms_prev": 3.0 if offset % 3 == 0 else 0.0, "obs_last3": pm,
                "obs_last24": pm - 1, "obs_prevday": pm - 2,
                "hrrr_u10_L0": 2.0, "hrrr_v10_L0": 0.0,
                **{f"aqm_mean_L{lead}": pm + lead + 4 for lead in (0, 1)},
                **{f"y_L{lead}": pm + lead for lead in (0, 1)},
            })
    return pd.DataFrame(rows)


def test_backtest_exports_the_evaluated_ensemble_and_forecast(tmp_path, monkeypatch):
    points = pd.DataFrame({
        "point_id": ["A", "B", "C"], "kind": "epa",
        "lat": [38.0, 38.1, 38.2], "lon": [-121.0, -121.1, -121.2],
    })
    train = _synthetic_table(date(2021, 7, 1), 8, points)
    calibrate = _synthetic_table(date(2023, 7, 1), 21, points)
    cache = tmp_path / "processed"
    cache.mkdir()
    train.to_parquet(cache / "table_train.parquet", index=False)
    calibrate.to_parquet(cache / "table_calibrate.parquet", index=False)
    # No table_test exists: dev mode must never try to read the reserved split.
    monkeypatch.setattr(backtest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(backtest, "processed_dir", lambda study: cache)
    monkeypatch.setattr(backtest, "load_points", lambda name: points.copy())
    monkeypatch.setattr(backtest, "load_forecasts", lambda days: pd.DataFrame())
    real_trainer = backtest.Trainer
    trained = []

    def record_trainer(n_feat, n_leads, graph, feat_names, cfg):
        trained.append((cfg.use_graph, cfg.seed))
        return real_trainer(n_feat, n_leads, graph, feat_names, cfg)

    monkeypatch.setattr(backtest, "Trainer", record_trainer)
    study = Study({"name": "synthetic_pipeline", "leads": [0, 1],
                   "graph": {"k_neighbors": 2, "max_km": 300}})
    args = SimpleNamespace(final=False, tag="smoke", train="train", calibrate="calibrate",
                           test="test", seed=4, seeds=2, epochs=3, hidden=16, layers=1,
                           dropout=0.0, smoke_weight=1.0, no_gbm=True, no_ablation=False)

    result = backtest.run_backtest(study, args)

    assert trained == [(True, 4), (True, 5), (False, 4), (False, 5)]
    assert "smokecast_nograph" in set(result["metrics"].model)
    run = tmp_path / "outputs" / "backtest" / "synthetic_pipeline_smoke"
    model, prep, _cqr, metadata = infer.load_artifacts(run)
    assert metadata["n_members"] == 2
    assert metadata["seeds"] == [4, 5]
    assert not result["metrics"].empty
    assert not (run / "FINAL_RUN_LOCK.json").exists()
    assert "backtest (dev)" in (run / "report.md").read_text()
    test = calibrate[backtest._week_parity(calibrate) == 1].reset_index(drop=True)
    graph = build_knn_graph(points.lat, points.lon, k=2, max_km=300)
    gd = to_graph_data(test, prep, metadata["node_ids"])
    with torch.no_grad():
        q = model(torch.as_tensor(gd.X), torch.as_tensor(gd.wind), torch.as_tensor(gd.base),
                  torch.as_tensor(graph.src), torch.as_tensor(graph.dst),
                  torch.as_tensor(graph.bearing, dtype=torch.float32),
                  torch.as_tensor(graph.edge_attr)).numpy()
    assert q.shape == (len(gd.days), len(points), 2, 5)
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=-1) >= 0).all()
    expected = backtest._rows_from_graph(q, gd, len(test))
    evaluated = pd.read_parquet(run / "predictions.parquet")
    for lead in study.leads:
        np.testing.assert_allclose(evaluated[f"smokecast_L{lead}"],
                                   np.maximum(0, np.expm1(expected[:, lead, 2])), rtol=1e-5)

    day = gd.days[0]
    day_table = test[test.issue_date == day].reset_index(drop=True)
    monkeypatch.setattr(infer, "load_points", lambda name: points.copy())
    monkeypatch.setattr(infer, "build_table", lambda *args, **kwargs: day_table.copy())
    forecast = infer.forecast_day(study, day, run, with_grid=False)
    assert len(forecast) == len(points) * len(study.leads)
    quantiles = forecast[["q05", "q25", "q50", "q75", "q95"]].to_numpy()
    assert np.isfinite(quantiles).all()
    assert (quantiles >= 0).all()
    assert (np.diff(quantiles, axis=-1) >= 0).all()
    for lead in study.leads:
        live = forecast[forecast.lead == lead].set_index("point_id")
        scored = evaluated[evaluated.issue_date == day].set_index("point_id").loc[live.index]
        for forecast_col, scored_col in (("q50", f"smokecast_L{lead}"),
                                         ("q05", f"smokecast_lo_L{lead}"),
                                         ("q95", f"smokecast_hi_L{lead}")):
            np.testing.assert_allclose(live[forecast_col], scored[scored_col], rtol=1e-5)
