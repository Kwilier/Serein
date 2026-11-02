"""Gradient-boosted trees: the strong "no graph learning" comparison (TODO B-5).

Same inputs as the GNN, plus hand-made neighbor averages (inverse-distance
weighted). If the GNN beats this, the learned graph is doing real work.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

GBM_QUANTILES = (0.05, 0.5, 0.95)


class QuantileGBM:
    def __init__(self, quantiles=GBM_QUANTILES, max_iter: int = 400, seed: int = 0):
        self.quantiles = quantiles
        self.max_iter, self.seed = max_iter, seed
        self.models: dict = {}

    def fit(self, X, y_delta, X_val=None, y_val=None):
        """X [rows, F]; y_delta [rows, L] = target - base (log1p space)."""
        for lead in range(y_delta.shape[1]):
            ok = np.isfinite(y_delta[:, lead])
            for q in self.quantiles:
                m = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=self.max_iter, learning_rate=0.05,
                    max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                    early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
                    random_state=self.seed)
                m.fit(X[ok], y_delta[ok, lead])
                self.models[(lead, q)] = m
        return self

    def predict(self, X, n_leads: int) -> np.ndarray:
        """[rows, L, Q] deltas, sorted along Q so quantiles never cross."""
        out = np.zeros((len(X), n_leads, len(self.quantiles)), np.float32)
        for lead in range(n_leads):
            for j, q in enumerate(self.quantiles):
                out[:, lead, j] = self.models[(lead, q)].predict(X)
        return np.sort(out, axis=-1)
