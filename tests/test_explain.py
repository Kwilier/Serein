"""Explainability helpers (B-11)."""

import torch

from serein.eval.explain import feature_group, integrated_gradients


def test_feature_groups():
    assert feature_group("obs_last24") == "live monitor readings"
    assert feature_group("obs_last24__missing") == "live monitor readings"
    assert feature_group("hrrr_pm_L1") == "HRRR smoke forecast"
    assert feature_group("hrrr_t2m_L1") == "HRRR weather"
    assert feature_group("aqm_mean_L2") == "NOAA AQM forecast"
    assert feature_group("frp_upwind_250") == "satellite fire detections"
    assert feature_group("hms_prev") == "smoke plume maps (HMS)"
    assert feature_group("doy_sin") == "place and season"


def test_integrated_gradients_add_up_to_the_change():
    w = torch.tensor([0.5, -2.0, 1.0])
    f = lambda x: torch.tanh(x @ w) + (x[1] * x[2])   # nonlinear, with an interaction
    x, x0 = torch.tensor([1.0, 0.3, -0.7]), torch.zeros(3)
    a = integrated_gradients(f, x, x0, steps=200)
    assert abs(float(a.sum()) - float(f(x) - f(x0))) < 1e-3
