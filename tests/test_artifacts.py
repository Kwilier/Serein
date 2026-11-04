"""Inference must reconstruct the ensemble that was evaluated and calibrated."""

import json
import pickle

import pytest
import torch

from serein.features.graph import build_knn_graph
from serein.models.conformal import CQR
from serein.models.ensemble import SmokeCastEnsemble
from serein.models.gnn import SmokeCastGNN
from serein.models.infer import load_artifacts, save_artifacts
from serein.models.train import TrainConfig


def _members(count=3, use_graph=True):
    cfg = TrainConfig(hidden=8, layers=1, heads=2, seed=12, use_graph=use_graph)
    members = []
    for seed in range(cfg.seed, cfg.seed + count):
        torch.manual_seed(seed)
        members.append(SmokeCastGNN(2, 1, cfg.hidden, cfg.layers, cfg.heads,
                                   drop=0.0, use_graph=cfg.use_graph).eval())
    return cfg, members


def _inputs():
    graph = build_knn_graph([38.0, 38.1, 38.2], [-121.0, -121.1, -121.2], k=2)
    generator = torch.Generator().manual_seed(30)
    return (
        torch.randn(2, 3, 2, generator=generator),
        torch.randn(2, 3, 2, generator=generator),
        torch.ones(2, 3, 1), torch.as_tensor(graph.src), torch.as_tensor(graph.dst),
        torch.as_tensor(graph.bearing, dtype=torch.float32), torch.as_tensor(graph.edge_attr),
    )


def _export(path, count=3, use_graph=True):
    cfg, members = _members(count, use_graph)
    calibration = CQR(qhat={(0, -1): 0.25})
    save_artifacts(path, {"features": ["obs", "wind"]}, ["A", "B", "C"], 2, [1], cfg,
                   calibration, [member.state_dict() for member in members],
                   seeds=list(range(cfg.seed, cfg.seed + count)))
    return cfg, members


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("use_graph", [True, False])
def test_export_loads_every_member_and_averages_evaluated_quantiles(tmp_path, count, use_graph):
    cfg, members = _export(tmp_path, count, use_graph)
    model, prep, calibration, metadata = load_artifacts(tmp_path)
    inputs = _inputs()
    with torch.no_grad():
        expected = torch.stack([member(*inputs) for member in members]).mean(0)
        actual = model(*inputs)

    torch.testing.assert_close(actual, expected)
    if count > 1:
        assert not torch.allclose(actual, members[0](*inputs))
    assert prep == {"features": ["obs", "wind"]}
    assert calibration.qhat == {(0, -1): 0.25}
    assert metadata["n_members"] == count
    assert metadata["seeds"] == list(range(cfg.seed, cfg.seed + count))
    assert metadata["aggregation"] == "mean_log1p_quantiles"
    assert all(not member.training for member in model.members)
    assert model.use_graph is use_graph


def test_single_state_dict_export_keeps_existing_caller_api(tmp_path):
    cfg, members = _members(count=1)
    save_artifacts(tmp_path, None, ["A"], 2, [1], cfg, CQR(), members[0].state_dict())
    model, _, _, metadata = load_artifacts(tmp_path)
    assert metadata["n_members"] == 1
    assert metadata["seeds"] == [cfg.seed]
    torch.testing.assert_close(model(*_inputs()), members[0](*_inputs()))


def test_legacy_single_member_artifact_loads_with_explicit_warning(tmp_path):
    cfg, members = _members(count=1)
    torch.save(members[0].state_dict(), tmp_path / "smokecast.pt")
    (tmp_path / "model.json").write_text(json.dumps({
        "node_ids": ["A", "B", "C"], "n_feat": 2, "leads": [1],
        "hidden": cfg.hidden, "layers": cfg.layers, "heads": cfg.heads,
    }))
    with (tmp_path / "artifacts.pkl").open("wb") as file:
        pickle.dump({"prep": None, "cqr": CQR()}, file)

    with pytest.warns(UserWarning, match="legacy single-model"):
        model, _, _, metadata = load_artifacts(tmp_path)
    torch.testing.assert_close(model(*_inputs()), members[0](*_inputs()))
    assert metadata["n_members"] == 1


def test_exported_ensemble_preserves_input_gradients_and_mean_attention(tmp_path):
    _, members = _export(tmp_path)
    model, *_ = load_artifacts(tmp_path)
    inputs = list(_inputs())
    inputs[0].requires_grad_(True)

    expected = torch.stack([member(*inputs) for member in members]).mean(0)
    expected_grad, = torch.autograd.grad(expected[..., 2].sum(), inputs[0])
    expected_attention = torch.stack([member.last_attention for member in members]).mean(0)
    actual = model(*inputs)
    actual_grad, = torch.autograd.grad(actual[..., 2].sum(), inputs[0])

    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad.abs().sum() > 0
    torch.testing.assert_close(model.last_attention, expected_attention)
    assert model.last_attention.requires_grad is False
    assert (model.nq, model.n_leads) == (5, 1)


@pytest.mark.parametrize("changes", [
    {"format_version": 3}, {"format_version": 2.0}, {"format_version": True},
    {"n_members": 0}, {"n_members": True}, {"n_feat": 0},
    {"leads": []}, {"leads": [1, 1]}, {"leads": [True]},
    {"node_ids": ["A", "A"]}, {"node_ids": [None]},
    {"hidden": 7}, {"heads": 0}, {"use_graph": "false"},
    {"seeds": [12]}, {"aggregation": "median"},
])
def test_invalid_metadata_is_rejected_clearly(tmp_path, changes):
    _export(tmp_path)
    path = tmp_path / "model.json"
    metadata = json.loads(path.read_text())
    metadata.update(changes)
    path.write_text(json.dumps(metadata))
    with pytest.raises((TypeError, ValueError)):
        load_artifacts(tmp_path)


@pytest.mark.parametrize("weights", [
    {}, {"format_version": 2, "members": []},
    {"format_version": 2, "members": [{}]},
    {"format_version": 2, "members": [None, None, None]},
])
def test_mismatched_or_broken_weights_are_rejected_clearly(tmp_path, weights):
    _export(tmp_path)
    torch.save(weights, tmp_path / "smokecast.pt")
    with pytest.raises(ValueError, match="checkpoint"):
        load_artifacts(tmp_path)


def test_empty_ensemble_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SmokeCastEnsemble([])
