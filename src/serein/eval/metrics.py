"""Scoring rules. All inputs in µg/m³.

RMSE / MAE / bias      size of errors (RMSE punishes big misses more)
skill vs. X            1 - RMSE_model / RMSE_X  (0.20 = 20% smaller errors than X)
category accuracy      % of days the AQI color is exactly right
USG+ POD               of days that really were >= 35.5 µg/m³ (orange or worse),
                       what % did we warn about? (probability of detection)
USG+ FAR               of our warnings, what % were false alarms?
CSI                    hits / (hits + misses + false alarms): one number for both
CRPS                   accuracy of the whole forecast distribution (= MAE for a
                       single-number forecast, so it compares fairly with baselines)
coverage90 / width90   how often truth fell in the 90% range, and how wide it was
"""

from __future__ import annotations

import numpy as np

from serein.aqi import USG_THRESHOLD, pm25_to_category


def point_metrics(pred, obs, dates=None) -> dict:
    pred, obs = np.asarray(pred), np.asarray(obs)
    if pred.ndim != 1 or obs.ndim != 1 or pred.shape != obs.shape:
        raise ValueError("predictions and observations must have the same one-dimensional shape")
    if dates is not None:
        dates = np.asarray(dates)
        if dates.shape != obs.shape:
            raise ValueError("dates must have the same shape as observations")
    ok = np.isfinite(pred) & np.isfinite(obs)
    p, o = pred[ok], obs[ok]
    if len(o) == 0:
        return {"n": 0}
    err = p - o
    out = {
        "n": len(o),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(p, o)[0, 1]) if len(o) > 2 and np.std(p) > 0 else np.nan,
    }
    if dates is not None:
        dd = np.asarray(dates)[ok]
        cp = np.concatenate([pm25_to_category(p[dd == d], d) for d in np.unique(dd)])
        co = np.concatenate([pm25_to_category(o[dd == d], d) for d in np.unique(dd)])
    else:
        cp, co = pm25_to_category(p), pm25_to_category(o)
    out["cat_acc"] = float(np.mean(cp == co))
    out["cat_within1"] = float(np.mean(np.abs(cp - co) <= 1))
    hit = np.sum((p >= USG_THRESHOLD) & (o >= USG_THRESHOLD))
    miss = np.sum((p < USG_THRESHOLD) & (o >= USG_THRESHOLD))
    fa = np.sum((p >= USG_THRESHOLD) & (o < USG_THRESHOLD))
    out["usg_events"] = int(hit + miss)
    out["usg_pod"] = float(hit / (hit + miss)) if hit + miss else np.nan
    out["usg_far"] = float(fa / (hit + fa)) if hit + fa else np.nan
    out["usg_csi"] = float(hit / (hit + miss + fa)) if hit + miss + fa else np.nan
    return out


def crps_from_quantiles(qs: np.ndarray, levels, obs: np.ndarray) -> np.ndarray:
    """Approximate CRPS = 2 * mean pinball loss over quantile levels. qs [rows, Q]."""
    qs, levels, obs = np.asarray(qs), np.asarray(levels), np.asarray(obs)
    if (qs.ndim != 2 or obs.ndim != 1 or levels.ndim != 1
            or qs.shape != (obs.size, levels.size) or levels.size == 0):
        raise ValueError("quantile shape must be [rows, levels] with one observation per row")
    if (not np.isfinite(levels).all() or np.any((levels < 0) | (levels > 1))
            or np.any(np.diff(levels) <= 0)):
        raise ValueError("quantile levels must be finite, strictly increasing probabilities in [0, 1]")
    if np.any(np.diff(qs, axis=1) < 0):
        raise ValueError("quantiles must be nondecreasing along probability levels")
    lv = levels[None, :]
    d = obs[:, None] - qs
    return 2 * np.mean(np.maximum(lv * d, (lv - 1) * d), axis=1)


def interval_metrics(lo, hi, obs) -> dict:
    lo, hi, obs = np.asarray(lo), np.asarray(hi), np.asarray(obs)
    if lo.shape != hi.shape or lo.shape != obs.shape:
        raise ValueError("interval bounds and observations must have the same shape")
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(obs)
    if not ok.any():
        return {}
    if np.any(lo[ok] > hi[ok]):
        raise ValueError("interval lower bounds must not exceed upper bounds")
    inside = (obs[ok] >= lo[ok]) & (obs[ok] <= hi[ok])
    return {"coverage90": float(inside.mean()), "width90": float(np.median(hi[ok] - lo[ok]))}


def bootstrap_rmse_diff(pred_a, pred_b, obs, dates, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CI for RMSE(a) - RMSE(b), resampling whole DAYS (errors on one day are
    correlated across sites, so resampling rows would overstate confidence)."""
    if (isinstance(n_boot, (bool, np.bool_))
            or not isinstance(n_boot, (int, np.integer)) or n_boot <= 0):
        raise ValueError("n_boot must be a positive integer")
    pred_a, pred_b, obs, dates = map(np.asarray, (pred_a, pred_b, obs, dates))
    if (obs.ndim != 1 or any(a.shape != obs.shape for a in (pred_a, pred_b, dates))):
        raise ValueError("predictions, observations and dates must have the same one-dimensional shape")
    ok = np.isfinite(pred_a) & np.isfinite(pred_b) & np.isfinite(obs)
    a2, b2 = (pred_a[ok] - obs[ok]) ** 2, (pred_b[ok] - obs[ok]) ** 2
    dd = np.asarray(dates)[ok]
    days, inv = np.unique(dd, return_inverse=True)
    if len(days) < 5:
        return {}
    sa = np.bincount(inv, a2, len(days))
    sb = np.bincount(inv, b2, len(days))
    cnt = np.bincount(inv, minlength=len(days))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(days), len(days))
        n = cnt[k].sum()
        diffs[i] = np.sqrt(sa[k].sum() / n) - np.sqrt(sb[k].sum() / n)
    point = np.sqrt(a2.mean()) - np.sqrt(b2.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_a_better": float(np.mean(diffs < 0))}
