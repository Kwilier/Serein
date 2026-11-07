"""Exercise optional fire inputs through preprocessing, fitting and reload."""

import numpy as np
import pandas as pd
import torch

from serein.features.fire_history import augment_fire_history
from serein.features.graph import build_knn_graph
from serein.features.prepare import Prep, to_graph_data
from serein.models.train import TrainConfig, Trainer


def test_fire_history_survives_training_missingness_and_model_reload(tmp_path, monkeypatch):
    points = pd.DataFrame({"point_id": ["A", "B", "C"],
                           "lat": [38., 38.2, 38.4], "lon": [-121., -121.2, -121.4]})
    days = pd.date_range("2023-06-01", periods=14)
    fires, coverage = [], []
    for i, day in enumerate(days):
        end = day + pd.Timedelta(hours=15.5)
        fires.append({"source": "VIIRS", "latitude": 38., "longitude": -121.,
                      "frp": 10 + (i % 3) * 40., "time_utc": end - pd.Timedelta(hours=2),
                      "available_utc": end - pd.Timedelta(hours=1)})
        if i != 5:
            coverage.append({"source": "VIIRS", "start_utc": end - pd.Timedelta(days=1),
                             "end_utc": end, "available_utc": end + pd.Timedelta(hours=3),
                             "lat_min": 30., "lat_max": 50., "lon_min": -130., "lon_max": -110.})
    table = pd.concat([points.assign(issue_date=day.date(), kind="epa", hms_prev=0.,
                                     obs_prevday=12., obs_last24=15.,
                                     y_L0=15 + i, y_L1=17 + i)
                       for i, day in enumerate(days[2:])], ignore_index=True)
    augmented = augment_fire_history(table, pd.DataFrame(fires), pd.DataFrame(coverage),
                                     source="VIIRS", issue_hour_utc=18.5,
                                     acquisition_lag_hours=3)
    train = augmented[augmented.issue_date < days[10].date()].copy()
    validation = augmented[augmented.issue_date >= days[10].date()].copy()
    prep = Prep().fit(train)
    transformed, names = prep.transform(train)
    assert "fire_recent_frp__missing" in names
    assert np.isfinite(transformed).all()
    column = "fire_frp_log_change"
    finite = train[column].notna().to_numpy()
    restored = transformed[finite, names.index(column)] * prep.std[column] + prep.mean[column]
    np.testing.assert_allclose(restored, train.loc[finite, column], atol=1e-6)
    assert (restored < 0).any() and (restored > 0).any()
    ablated = Prep(exclude=("fire_",)).fit(train)
    assert not any(name.startswith("fire_") for name in ablated.feature_cols)

    nodes = points.point_id.tolist()
    graph = build_knn_graph(points.lat, points.lon, k=2, max_km=300)
    training = to_graph_data(train, prep, nodes)
    checking = to_graph_data(validation, prep, nodes)
    monkeypatch.setattr("serein.models.train._device", lambda: torch.device("cpu"))
    config = TrainConfig(hidden=16, layers=1, dropout=0., epochs=3, seed=7)
    trainer = Trainer(training.X.shape[-1], 2, graph, training.feat_names, config)
    result = trainer.fit(training, checking, log=lambda line: None)
    assert np.isfinite(result["best_val"])
    predicted = trainer.predict(checking)
    assert predicted.shape == (4, 3, 2, 5)
    assert np.isfinite(predicted).all() and (np.diff(predicted, axis=-1) >= 0).all()
    checkpoint = tmp_path / "model.pt"
    torch.save(trainer.model.state_dict(), checkpoint)
    reloaded = Trainer(training.X.shape[-1], 2, graph, training.feat_names, config)
    reloaded.model.load_state_dict(torch.load(checkpoint, weights_only=True))
    np.testing.assert_array_equal(reloaded.predict(checking), predicted)
