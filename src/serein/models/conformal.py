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
        if bins.ndim != 1 or not np.isfinite(bins).all() or not (np.diff(bins) > 0).all():
            raise ValueError("bins must be finite, one-dimensional, and strictly increasing")

    def group(self, median: np.ndarray) -> np.ndarray:
        self._validate_settings()
        return np.digitize(median, self.bins)

    def fit(self, lo, hi, median, y) -> CQR:
        """Arrays [rows, L] in log1p space."""
        _validate_shapes(lo, hi, median, y)
        self._validate_settings()
        L = lo.shape[1]
        self.qhat.clear()
        for lead in range(L):
            g = self.group(median[:, lead])
            s = np.maximum(lo[:, lead] - y[:, lead], y[:, lead] - hi[:, lead])
            self.qhat[(lead, -1)] = _qhat(s, self.alpha)
            for k in np.unique(g):
                sk = s[g == k]
                # Fall back to the pooled value when a group is tiny.
                if np.isfinite(sk).sum() >= 50:
                    self.qhat[(lead, int(k))] = _qhat(sk, self.alpha)
        return self

    def apply(self, lo, hi, median):
        _validate_shapes(lo, hi, median)
        self._validate_settings()
        lo2 = lo.astype(np.result_type(lo.dtype, np.float32), copy=True)
        hi2 = hi.astype(np.result_type(hi.dtype, np.float32), copy=True)
        n_groups = len(self.bins) + 1
        for lead in range(lo.shape[1]):
            pooled = self.qhat.get((lead, -1), 0.0)
            table = np.array([self.qhat.get((lead, k), pooled) for k in range(n_groups)])
            q = table[self.group(median[:, lead])]
            lo2[:, lead] = lo[:, lead] - q
            hi2[:, lead] = hi[:, lead] + q
        return lo2, hi2
