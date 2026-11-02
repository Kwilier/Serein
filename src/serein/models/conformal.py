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


def _qhat(scores: np.ndarray, alpha: float) -> float:
    s = scores[np.isfinite(scores)]
    n = len(s)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


@dataclass
class CQR:
    alpha: float = 0.10
    bins: tuple = (np.log1p(12.0), np.log1p(35.5))  # group by predicted median (log1p)
    qhat: dict = field(default_factory=dict)        # (lead, group) -> Q

    def group(self, median: np.ndarray) -> np.ndarray:
        return np.digitize(median, self.bins)

    def fit(self, lo, hi, median, y) -> CQR:
        """Arrays [rows, L] in log1p space."""
        L = lo.shape[1]
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
        lo2, hi2 = lo.copy(), hi.copy()
        for lead in range(lo.shape[1]):
            g = self.group(median[:, lead])
            q = np.array([self.qhat.get((lead, int(k)), self.qhat.get((lead, -1), 0.0))
                          for k in g])
            lo2[:, lead] = lo[:, lead] - q
            hi2[:, lead] = hi[:, lead] + q
        return lo2, hi2
