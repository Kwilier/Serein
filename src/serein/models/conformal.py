"""Conformal calibration: make "90% range" mean 90%.

Neural networks are often over-confident. Conformalized Quantile Regression
(CQR; Romano, Patterson & Candès, NeurIPS 2019) fixes this using days the model
never trained on:

1. On calibration days, measure how far each observation fell outside the
   predicted [5%, 95%] range (negative if inside):  s = max(lo - y, y - hi).
2. Take the 90th percentile of those scores (with a small finite-sample
   correction) = Q.
3. Widen (or shrink, if Q < 0) every future range by Q:  [lo - Q, hi + Q].

"Mondrian" version: do this separately per group (e.g. clean vs. smoky
forecasts), so the range is honest on smoke days too, not just on average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _validate_shapes(*arrays):
    shape = arrays[0].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays[1:]):
        raise ValueError("Calibration arrays must have the same [rows, leads] shape")


def _validate_bounds(lo, hi):
    if np.any(lo > hi):
        raise ValueError("Lower forecast bounds must not exceed upper bounds")


def _qhat(scores: np.ndarray, alpha: float) -> float:
    s = scores[np.isfinite(scores)]
    n = len(s)
    if n == 0:
        return 0.0
    rank = max(1, min(n, int(np.ceil((n + 1) * (1 - alpha)))))
    return float(np.partition(s, rank - 1)[rank - 1])


@dataclass
class CQR:
    alpha: float = 0.10
    bins: tuple = (np.log1p(12.0), np.log1p(35.5))  # group by predicted median (log1p)
    qhat: dict = field(default_factory=dict)        # (lead, group) -> Q

    def _validate_settings(self):
        if not np.isscalar(self.alpha) or not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be finite and between 0 and 1")
        bins = np.asarray(self.bins)
        if bins.ndim != 1 or not np.isfinite(bins).all() or not (bins[1:] > bins[:-1]).all():
            raise ValueError("bins must be finite, one-dimensional, and strictly increasing")

    def group(self, median: np.ndarray) -> np.ndarray:
        self._validate_settings()
        return np.digitize(median, self.bins)

    def fit(self, lo, hi, median, y) -> CQR:
        """Arrays [rows, L] in log1p space."""
        _validate_shapes(lo, hi, median, y)
        self._validate_settings()
        _validate_bounds(lo, hi)
        dtype = np.result_type(lo.dtype, hi.dtype, y.dtype, np.float32)
        lo, hi, y = (array.astype(dtype, copy=False) for array in (lo, hi, y))
        L = lo.shape[1]
        self.qhat.clear()
        for lead in range(L):
            g = self.group(median[:, lead])
            s = np.maximum(lo[:, lead] - y[:, lead], y[:, lead] - hi[:, lead])
            self.qhat[(lead, -1)] = _qhat(s, self.alpha)
            for k in np.unique(g):
                sk = s[(g == k) & np.isfinite(median[:, lead])]
                # Fall back to the pooled value when a group is tiny.
                if np.isfinite(sk).sum() >= 50:
                    self.qhat[(lead, int(k))] = _qhat(sk, self.alpha)
        return self

    def apply(self, lo, hi, median):
        _validate_shapes(lo, hi, median)
        self._validate_settings()
        _validate_bounds(lo, hi)
        lo2 = lo.astype(np.result_type(lo.dtype, np.float32), copy=True)
        hi2 = hi.astype(np.result_type(hi.dtype, np.float32), copy=True)
        n_groups = len(self.bins) + 1
        for lead in range(lo.shape[1]):
            pooled = self.qhat.get((lead, -1), 0.0)
            table = np.array([self.qhat.get((lead, k), pooled) for k in range(n_groups)])
            q = table[self.group(median[:, lead])]
            q = np.where(np.isfinite(median[:, lead]), q, pooled)
            lo2[:, lead] = lo[:, lead] - q
            hi2[:, lead] = hi[:, lead] + q
        return lo2, hi2


def calibrate_quantiles(q_log: np.ndarray, cqr: CQR, levels) -> np.ndarray:
    """Calibrate log1p PM quantiles while preserving their order and physical support.

    CQR may shrink the outer interval past the inner quantiles. Keep the median,
    constrain the other quantiles to the calibrated interval, and bound log1p PM
    below by zero. Missing forecast rows remain missing. Use this same operation
    in evaluation, exported maps, and transfer scoring.
    """
    q = np.asarray(q_log, dtype=float)
    lv = np.asarray(levels, dtype=float)
    if (q.ndim != 3 or lv.ndim != 1 or q.shape[-1] != len(lv) or len(lv) < 3
            or not np.isfinite(lv).all() or (lv <= 0).any() or (lv >= 1).any()
            or (np.diff(lv) <= 0).any() or np.count_nonzero(lv == 0.5) != 1
            or not lv[0] < 0.5 < lv[-1]):
        raise ValueError("quantiles must have [rows, leads, levels] shape and ordered "
                         "probabilities including the median")
    if np.isinf(q).any() or (np.diff(q, axis=-1) < 0).any():
        raise ValueError("forecast quantiles must be ordered and finite or missing")
    mid = int(np.flatnonzero(lv == 0.5)[0])
    lo, hi = cqr.apply(q[..., 0], q[..., -1], q[..., mid])
    median = np.maximum(q[..., mid], 0.0)
    lower = np.maximum(0.0, np.minimum(lo, median))
    upper = np.maximum(hi, median)
    out = q.copy()
    out[..., :mid] = np.clip(q[..., :mid], lower[..., None], median[..., None])
    out[..., mid] = median
    out[..., mid + 1:] = np.clip(q[..., mid + 1:], median[..., None], upper[..., None])
    out[..., 0], out[..., -1] = lower, upper
    return out
