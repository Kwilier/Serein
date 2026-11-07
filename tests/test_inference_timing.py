import hashlib
import json
from copy import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from serein.eval.inference_timing import benchmark_forward, main, profile_export
from serein.features.graph import build_knn_graph
from serein.features.prepare import Prep, to_graph_data
from serein.models.conformal import CQR, calibrate_quantiles
from serein.models.ensemble import SmokeCastEnsemble
from serein.models.gnn import QUANTILES, SmokeCastGNN
from serein.models.infer import save_artifacts
from serein.models.train import TrainConfig


@pytest.fixture
def sample(tmp_path):
    points = pd.DataFrame({"point_id": ["A", "B", "C"], "lat": [38., 38.1, 38.2],
                           "lon": [-121., -121.1, -121.2]})
    table = pd.concat([points.assign(issue_date=day.date(), kind="epa", hms_prev=0.,
                                     obs_last24=10 + i, obs_prevday=11 + i,
                                     y_L0=12 + i, y_L1=13 + i)
                       for i, day in enumerate(pd.date_range("2023-07-01", periods=4))],
                      ignore_index=True)
    prep = Prep().fit(table)
    daily = table[table.issue_date == pd.Timestamp("2023-07-03").date()]
    data = to_graph_data(daily, prep, points.point_id.tolist())
    graph = build_knn_graph(points.lat, points.lon, k=2, max_km=300)
    torch.manual_seed(71)
    members = [SmokeCastGNN(data.X.shape[-1], 2, hidden=16, layers=1, drop=0.) for _ in range(2)]
    ensemble = SmokeCastEnsemble(members).eval()
    cqr = CQR(qhat={(0, -1): .2, (1, -1): .3})
    run = tmp_path / "run"
    save_artifacts(run, prep, points.point_id.tolist(), data.X.shape[-1], [0, 1],
                   TrainConfig(hidden=16, layers=1), cqr,
                   [member.state_dict() for member in members], seeds=[71, 72])
    return points, table, data, graph, ensemble, cqr, run


def test_profile_matches_direct_calibrated_forward_and_records_scope(sample):
    _, _, data, graph, model, cqr, _ = sample
    report, values = benchmark_forward(model, data, graph, cqr, repeats=3, warmup=1)
    with torch.no_grad():
        raw = model(torch.as_tensor(data.X), torch.as_tensor(data.wind), torch.as_tensor(data.base),
                    torch.as_tensor(graph.src), torch.as_tensor(graph.dst),
                    torch.as_tensor(graph.bearing, dtype=torch.float32),
                    torch.as_tensor(graph.edge_attr)).numpy()[0]
    expected = np.expm1(calibrate_quantiles(raw, cqr, QUANTILES))
    np.testing.assert_array_equal(values, expected)
    assert report["nodes"] == 3 and report["edges"] == 6
    assert report["ensemble_members"] == 2
    assert len(report["calibrated_forward_seconds"]) == 3
    assert report["median_seconds"] == np.median(report["calibrated_forward_seconds"])
    assert report["p90_seconds"] >= report["median_seconds"] >= 0
    assert report["max_abs_repeat_difference"] == 0
    assert report["cuda_peak_allocated_bytes"] is None
    assert report["quantiles_sha256"] == hashlib.sha256(values.tobytes()).hexdigest()


def test_export_profile_ignores_targets_and_leaves_table_unchanged(sample):
    points, table, _, _, _, _, run = sample
    before = table.copy(deep=True)
    report, values = profile_export(run, table, points, "2023-07-03", repeats=1, warmup=0,
                                    neighbors=2)
    altered = table.assign(y_L0="not an observation", y_L1=-99999.)
    _, other = profile_export(run, altered, points, "2023-07-03", repeats=1, warmup=0, neighbors=2)
    np.testing.assert_array_equal(values, other)
    pd.testing.assert_frame_equal(table, before)
    assert report["present_model_nodes"] == 3
    assert report["input_rows"] == 3
    assert report["artifact_format"] == 2
    assert report["artifact_load_seconds"] >= 0 and report["prepare_seconds"] >= 0


@pytest.mark.parametrize("option,value", [("repeats", 0), ("repeats", True), ("repeats", 1.5),
                                         ("warmup", -1), ("warmup", True)])
def test_invalid_iteration_budgets(sample, option, value):
    _, _, data, graph, model, cqr, _ = sample
    with pytest.raises(ValueError, match=option):
        benchmark_forward(model, data, graph, cqr, **{option: value})


def test_rejects_multiday_and_temporal_inputs(sample):
    _, _, data, graph, model, cqr, _ = sample
    with pytest.raises(ValueError, match="one nonempty"):
        benchmark_forward(model, replace(data, days=[1, 2]), graph, cqr)
    temporal = copy(data)
    temporal.hist = np.zeros((1, 3, 2, data.X.shape[-1]))
    with pytest.raises(ValueError, match="history inputs"):
        benchmark_forward(model, temporal, graph, cqr)


def test_rejects_invalid_forecast_instead_of_timing_success(sample):
    _, _, data, graph, model, cqr, _ = sample
    bad = data.X.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid calibrated"):
        benchmark_forward(model, replace(data, X=bad), graph, cqr, repeats=1)


def test_explicit_cuda_failure_never_falls_back_to_cpu(sample, monkeypatch):
    _, _, data, graph, model, cqr, _ = sample
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        benchmark_forward(model, data, graph, cqr, device="cuda", repeats=1)


@pytest.mark.parametrize("problem", ["missing_date", "duplicate_input", "duplicate_points",
                                     "missing_coordinates"])
def test_export_profile_rejects_ambiguous_inputs(sample, problem):
    points, table, _, _, _, _, run = sample
    day = "2023-07-03"
    if problem == "missing_date":
        day = "2023-12-31"
    elif problem == "duplicate_input":
        table = pd.concat([table, table.iloc[[6]]])
    elif problem == "duplicate_points":
        points = pd.concat([points, points.iloc[[0]]])
    else:
        points = points.iloc[1:]
    with pytest.raises(ValueError):
        profile_export(run, table, points, day, repeats=1)


def test_cli_emits_hashed_profile_and_preserves_existing_output(sample, tmp_path):
    points, table, _, _, _, _, run = sample
    table_path, points_path = tmp_path / "table.parquet", tmp_path / "points.parquet"
    table.to_parquet(table_path)
    points.to_parquet(points_path)
    output = tmp_path / "profile" / "cpu.json"
    args = ["--run", str(run), "--table", str(table_path), "--points", str(points_path),
            "--issue-date", "2023-07-03", "--repeats", "2", "--warmup", "0", "--out", str(output)]
    main(args)
    report = json.loads(output.read_text())
    assert report["inputs"]["table"]["sha256"] == hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert report["inputs"]["smokecast.pt"]["sha256"] == hashlib.sha256(
        (run / "smokecast.pt").read_bytes()).hexdigest()
    assert report["repeats"] == 2
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_profile_synchronizes_and_matches_cpu(sample):
    _, _, data, graph, model, cqr, _ = sample
    _, cpu = benchmark_forward(model, data, graph, cqr, repeats=1, warmup=0)
    report, cuda = benchmark_forward(model, data, graph, cqr, device="cuda", repeats=2, warmup=1)
    np.testing.assert_allclose(cuda, cpu, rtol=1e-5, atol=1e-5)
    assert report["device"] == "cuda"
    assert report["cuda_peak_allocated_bytes"] > 0
