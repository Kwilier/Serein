"""The fast (vectorized) versions must give the same answers as the old loops (Q-6)."""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from serein.data import firms
from serein.eval.backtest import prob_exceed
from serein.models.conformal import CQR


def _old_prob_exceed(q_log, levels, thr=35.5):
    t, lv = np.log1p(thr), np.asarray(levels, float)
    out = np.full(len(q_log), np.nan)
    for i, q in enumerate(q_log):
        if np.all(np.isfinite(q)):
            out[i] = 1 - np.interp(t, q, lv, left=lv[0] / 2, right=1 - (1 - lv[-1]) / 2)
    return out


def test_prob_exceed_matches_row_loop():
    rng = np.random.default_rng(0)
    levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    q = np.sort(rng.normal(np.log1p(30), 0.8, (5000, 5)), 1)
    q[:50, 2] = q[:50, 1]                        # tied quantiles
    q[50:60] = np.log1p(35.5)                    # all quantiles exactly at the threshold
    q[60:70, 3] = np.nan                         # missing
    np.testing.assert_allclose(prob_exceed(q, levels), _old_prob_exceed(q, levels),
                               atol=1e-12, equal_nan=True)


def test_cqr_apply_matches_row_loop():
    rng = np.random.default_rng(1)
    med = rng.normal(2.5, 1.0, (3000, 2))
    lo, hi, y = med - 0.5, med + 0.5, med + rng.normal(0, 0.6, med.shape)
    cqr = CQR().fit(lo, hi, med, y)
    cqr.qhat.pop((1, 2), None)                   # force the pooled fallback for one group
    lo2, hi2 = cqr.apply(lo, hi, med)
    for lead in range(2):
        g = cqr.group(med[:, lead])
        q = np.array([cqr.qhat.get((lead, int(k)), cqr.qhat[(lead, -1)]) for k in g])
        np.testing.assert_allclose(lo2[:, lead], lo[:, lead] - q)
        np.testing.assert_allclose(hi2[:, lead], hi[:, lead] + q)


def _old_fire_features(fires, points, issue_time, u, v):
    out = pd.DataFrame(0.0, index=range(len(points)),
                       columns=["frp_50", "frp_100", "frp_250", "frp_upwind_250", "fire_dist_km"])
    f = fires[(fires["time_utc"] > issue_time - timedelta(hours=24))
              & (fires["time_utc"] <= issue_time)]
    flat, flon, frp = f["latitude"].to_numpy(), f["longitude"].to_numpy(), f["frp"].to_numpy()
    for i, (la, lo) in enumerate(zip(points["lat"], points["lon"], strict=True)):
        d = firms.haversine_km(la, lo, flat, flon)
        out.at[i, "fire_dist_km"] = min(d.min(), 1000.0)
        for r in (50, 100, 250):
            out.at[i, f"frp_{r}"] = frp[d <= r].sum()
        near = d <= 250
        if np.isfinite(u[i]) and np.isfinite(v[i]) and near.any():
            b = firms.bearing_rad(flat[near], flon[near], la, lo)
            a = np.clip(np.cos(b - np.arctan2(u[i], v[i])), 0, None)
            out.at[i, "frp_upwind_250"] = (frp[near] * a / np.maximum(d[near], 10)).sum()
    return out


def test_fire_features_match_point_loop():
    rng = np.random.default_rng(2)
    t = datetime(2023, 8, 18, 18, 30)
    fires = pd.DataFrame({"latitude": rng.uniform(38, 46, 400),
                          "longitude": rng.uniform(-124, -116, 400),
                          "frp": rng.exponential(20, 400),
                          "time_utc": [t - timedelta(hours=float(h)) for h in rng.uniform(0, 40, 400)]})
    pts = pd.DataFrame({"lat": rng.uniform(37, 47, 300), "lon": rng.uniform(-125, -115, 300)})
    u, v = rng.normal(0, 4, 300), rng.normal(0, 4, 300)
    u[:20] = np.nan
    new = firms.fire_features(fires, pts, t, 24, u, v)
    old = _old_fire_features(fires, pts, t, u, v)
    pd.testing.assert_frame_equal(new[old.columns], old, check_exact=False, rtol=1e-9)


def test_contiguous_runs_split_seasons():
    days = [date(2021, 6, 1) + timedelta(days=i) for i in range(5)]
    days += [date(2022, 6, 1) + timedelta(days=i) for i in range(3)]
    assert firms.contiguous_runs(days) == [(date(2021, 6, 1), date(2021, 6, 5)),
                                           (date(2022, 6, 1), date(2022, 6, 3))]
