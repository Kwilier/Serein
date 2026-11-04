from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from serein.eval import transfer


@pytest.mark.parametrize("missing", [False, True])
def test_transfer_coverage_excludes_missing_intervals(monkeypatch, tmp_path, missing):
    tab = pd.DataFrame({
        "point_id": ["site", "site"], "kind": ["epa", "epa"],
        "issue_date": ["2023-06-01", "2023-06-02"], "y_L1": [10.0, 10.0],
    })
    points = pd.DataFrame({"point_id": ["site"], "lat": [40.0], "lon": [-75.0]})
    q = torch.log1p(torch.tensor([[[[5., 8., 10., 12., 15.]]],
                                [[[5., 8., 10., 12., 15.]]]]))
    q[1 if not missing else 0:] = torch.nan
    graph_data = SimpleNamespace(
        X=np.zeros((2, 1, 1)), wind=np.zeros((2, 1, 2)), base=np.zeros((2, 1, 1)),
        row_index=np.array([[0], [1]]),
    )
    calibration = SimpleNamespace(apply=lambda lo, hi, mid: (lo, hi))
    monkeypatch.setattr(transfer, "load_artifacts", lambda path: (
        lambda *args: q, None, calibration, None))
    monkeypatch.setattr(transfer.pd, "read_parquet", lambda path: tab.copy())
    monkeypatch.setattr(transfer, "load_points", lambda name: points)
    monkeypatch.setattr(transfer, "to_graph_data", lambda *args: graph_data)
    monkeypatch.setattr(transfer, "baselines", lambda *args: {})
    monkeypatch.setattr(transfer, "REPO_ROOT", tmp_path)
    study = SimpleNamespace(name="synthetic", raw={"graph": {"k_neighbors": 0, "max_km": 100}})
    monkeypatch.setattr(transfer, "processed_dir", lambda study: tmp_path)

    result = transfer.run_transfer(study, tmp_path / "run")

    if missing:
        assert "coverage90" not in result
    else:
        assert result.loc[0, "n"] == 1
        assert result.loc[0, "coverage90"] == pytest.approx(1.0)
        assert result.loc[0, "width90"] == pytest.approx(10.0)
