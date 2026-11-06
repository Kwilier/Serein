"""Shared input checks for saved episode diagnostics."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _load_predictions(source: Path) -> tuple[pd.DataFrame, str]:
    """Hash the bytes actually decoded, even if a saved run is replaced later."""
    payload = source.read_bytes()
    return pd.read_parquet(io.BytesIO(payload)), hashlib.sha256(payload).hexdigest()


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"{column} must contain finite values or missing data")
    return values


def _choices(values, name: str, *, integers: bool = False) -> list:
    values = list(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")
    for value in values:
        if integers:
            valid = (not isinstance(value, (bool, np.bool_))
                     and isinstance(value, (int, np.integer)) and value >= 0)
        else:
            valid = isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value))
        if not valid:
            raise ValueError(f"invalid {name}: {value!r}")
    return values


def _observations(predictions: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    parts = []
    for column in predictions:
        if match := re.fullmatch(r"y_L(\d+)", str(column)):
            parts.append(pd.DataFrame({
                "point_id": predictions.point_id.to_numpy(),
                "valid_date": days + pd.Timedelta(days=int(match[1])),
                "observed": _values(predictions, column),
            }))
    if not parts:
        raise ValueError("predictions require at least one y_L<lead> observation column")
    observations = pd.concat(parts, ignore_index=True)
    grouped = observations.groupby(["point_id", "valid_date"], sort=True, observed=True)["observed"]
    bounds = grouped.agg(["min", "max"])
    finite = bounds["min"].notna()
    if not np.isclose(bounds.loc[finite, "min"], bounds.loc[finite, "max"],
                      rtol=1e-6, atol=1e-6).all():
        raise ValueError("conflicting observations for the same point and valid date")
    # Choose an existing value deterministically, without averaging targets.
    return grouped.min().reset_index()
