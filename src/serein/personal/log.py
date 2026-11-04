"""The daily check-in (TODO C-9): a 10-second log that teaches AirPlan about one child.

Design rules
------------
* Six taps, no typing. Parents of sick kids are tired.
* Ask about TODAY only (memory of past days is unreliable).
* Everything stays on the family's device. Nothing is uploaded (docs/ETHICS.md).

Questions -> fields
  1. Any asthma symptoms today? none / mild / moderate / severe     -> symptoms 0-3
  2. Rescue (quick-relief) inhaler puffs today, NOT counting before
     exercise? 0 / 1-2 / 3-4 / 5+                                    -> puffs_bucket 0-3
  3. Woke up at night because of asthma? yes / no                    -> night_wake
  4. Time outdoors today? little (<1 h) / some (1-3 h) / a lot (3h+) -> outdoor 0.1/0.3/0.6
  5. Has a cold or is sick? yes / no                                 -> cold
  6. Took daily controller medicine? yes / no / doesn't have one     -> missed_controller

"Symptom day" (the model's outcome) = symptoms >= 1 OR any rescue puffs OR night waking.
This mirrors how asthma control is judged clinically (daytime symptoms, reliever
use, night waking) and should be confirmed with a clinician (TODO C-8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from serein.personal.model import FEATURES, ChildModel, design_row

OUTDOOR_FRAC = {0: 0.1, 1: 0.3, 2: 0.6}


@dataclass
class CheckIn:
    day: date
    symptoms: int            # 0 none, 1 mild, 2 moderate, 3 severe
    puffs_bucket: int        # 0: 0, 1: 1-2, 2: 3-4, 3: 5+
    night_wake: bool
    outdoor: int             # 0 little, 1 some, 2 a lot
    cold: bool
    missed_controller: bool  # False if took it OR has no controller

    def __post_init__(self):
        if not isinstance(self.day, date) or isinstance(self.day, datetime):
            raise TypeError("check-in day must be a calendar date")
        if not all(isinstance(value, (bool, np.bool_)) for value in
                   (self.night_wake, self.cold, self.missed_controller)):
            raise ValueError("check-in yes/no answers must be boolean")
        if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in (self.symptoms, self.puffs_bucket, self.outdoor))
                or self.symptoms not in range(4) or self.puffs_bucket not in range(4)
                or self.outdoor not in OUTDOOR_FRAC):
            raise ValueError("check-in answer out of range")

    @property
    def symptom_day(self) -> int:
        return int(self.symptoms >= 1 or self.puffs_bucket >= 1 or self.night_wake)

    @property
    def needs_doctor_flag(self) -> bool:
        """Severe symptoms or 5+ puffs: the app shows 'call your doctor', not a forecast."""
        return self.symptoms >= 3 or self.puffs_bucket >= 3


def to_frame(entries: list[CheckIn]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(e) for e in entries], columns=[f.name for f in fields(CheckIn)])
    if df.empty:
        df["symptom_day"] = pd.Series(dtype=int)
        return df
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df["symptom_day"] = [e.symptom_day for e in entries]
    return df.drop_duplicates("day", keep="last").sort_values("day").reset_index(drop=True)


def model_rows(log: pd.DataFrame, pm_by_day: dict) -> tuple[np.ndarray, np.ndarray]:
    """Join the log with OBSERVED daily PM2.5 near home -> (X, y) for ChildModel.fit.

    Log days may be calendar dates or ISO date strings from save_local CSV files.
    The last entry for each day wins, matching to_frame when a check-in is edited.
    pm_by_day: {date: µg/m³}. Missing or nonfinite PM on today or either prior day
    excludes the row.
    """
    X, y = [], []
    if not log.empty:
        log = log.copy()
        log["day"] = log["day"].map(
            lambda day: date.fromisoformat(day) if isinstance(day, str) else day)
        log = log.drop_duplicates("day", keep="last")
        for column in ("cold", "missed_controller"):
            if not log[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError(f"check-in {column} answers must be boolean")
    for r in log.itertuples():
        d = r.day
        pm0 = pm_by_day.get(d)
        pm1 = pm_by_day.get(d - pd.Timedelta(days=1).to_pytimedelta())
        pm2 = pm_by_day.get(d - pd.Timedelta(days=2).to_pytimedelta())
        if any(pd.isna(pm) or not np.isfinite(pm) for pm in (pm0, pm1, pm2)):
            continue
        X.append(design_row(pm0, pm1, pm2, OUTDOOR_FRAC[r.outdoor], int(r.cold),
                            int(r.missed_controller)))
        y.append(r.symptom_day)
    if not X:
        return np.zeros((0, len(FEATURES))), np.zeros(0)
    return np.array(X), np.array(y, float)


def fit_child(log: pd.DataFrame, pm_by_day: dict) -> ChildModel:
    X, y = model_rows(log, pm_by_day)
    m = ChildModel()
    return m.fit(X, y) if len(y) else m


def save_local(log: pd.DataFrame, path: Path) -> None:
    """Save on THIS device only. Never commit or upload real logs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
