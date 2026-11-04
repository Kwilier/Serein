"""Turn hourly data into local-calendar-day values.

The AQI uses the 24-hour average from midnight to midnight LOCAL STANDARD TIME.
Each point has a ``gmt_offset`` (e.g. -8 for Pacific). Local hour = UTC + offset.

Example: a Sacramento site (offset -8). Local day 2024-09-02 covers UTC
2024-09-02 08:00 .. 2024-09-03 07:59.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HOURS_DAILY = 18  # EPA 75% completeness rule


def add_local_date(df: pd.DataFrame, offsets: pd.Series, time_col: str = "time_utc",
                   id_col: str = "point_id") -> pd.DataFrame:
    """Apply fixed standard-time offsets; naive timestamps are interpreted as UTC."""
    off = df[id_col].map(offsets)
    if not np.isfinite(off).all():
        raise ValueError("Each point must have a finite standard-time offset")
    utc = pd.to_datetime(df[time_col], utc=True, format="mixed").dt.tz_localize(None)
    if utc.isna().any():
        raise ValueError(f"{time_col} must not contain missing timestamps")
    local = utc + pd.to_timedelta(off, unit="h")
    df = df.assign(local_date=local.dt.date, _off=off)
    return df


def obs_daily_means(hourly: pd.DataFrame, offsets: pd.Series,
                    min_hours: int = MIN_HOURS_DAILY) -> pd.DataFrame:
    """Daily mean PM2.5; require min_hours distinct finite hours per local day.

    Readings within each local standard-time hour are averaged before daily aggregation so
    each hour has equal weight and counts only once toward completeness.
    """
    if (isinstance(min_hours, (bool, np.bool_))
            or not isinstance(min_hours, (int, np.integer)) or min_hours <= 0):
        raise ValueError("min_hours must be a positive integer")
    h = hourly.rename(columns={"site_id": "point_id"})
    h = h[h["point_id"].isin(offsets.index)]
    if h.empty:
        return pd.DataFrame({
            "point_id": h["point_id"].reset_index(drop=True),
            "local_date": pd.Series(dtype="object"),
            "pm25_daily": pd.Series(dtype="float64"),
            "n_hours": pd.Series(dtype="int64"),
        })
    h = add_local_date(h, offsets)
    utc = pd.to_datetime(h["time_utc"], utc=True, format="mixed").dt.tz_localize(None)
    h = h.assign(_local_hour=(utc + pd.to_timedelta(h["_off"], unit="h")).dt.floor("h"))
    h["pm25"] = h["pm25"].where(np.isfinite(h["pm25"]))
    h = h.groupby(["point_id", "local_date", "_local_hour"], as_index=False)["pm25"].mean()
    g = h.groupby(["point_id", "local_date"])["pm25"].agg(
        mean="mean", size="count").reset_index()
    g.loc[g["size"] < min_hours, "mean"] = np.nan
    return g.rename(columns={"mean": "pm25_daily", "size": "n_hours"})


def model_daily(df: pd.DataFrame, offsets: pd.Series, value_col: str = "value",
                min_count: int = 12, group_extra: tuple = ()) -> pd.DataFrame:
    """Aggregate a forecast (issue_date, point_id, valid_utc, value) to lead-day mean/max.

    lead = local_date - issue_date (days). Size counts distinct finite timestamps.
    Repeated values at a timestamp are averaged within each forecast group first.
    Issue dates are normalized to calendar dates before grouping.
    """
    if (isinstance(min_count, (bool, np.bool_))
            or not isinstance(min_count, (int, np.integer)) or min_count <= 0):
        raise ValueError("min_count must be a positive integer")
    d = df.copy()
    if d.empty:
        g = d[["issue_date", "point_id", *group_extra]].copy()
        for column, dtype in (("lead", "int64"), ("mean", "float64"),
                              ("max", "float64"), ("size", "int64")):
            g[column] = pd.Series(index=g.index, dtype=dtype)
        return g
    issue_dates = [pd.to_datetime(value, format="mixed") for value in d["issue_date"]]
    d["issue_date"] = [pd.NaT if pd.isna(value) else value.date() for value in issue_dates]
    d["valid_utc"] = pd.to_datetime(d["valid_utc"], utc=True, format="mixed")
    if d["valid_utc"].isna().any():
        raise ValueError("valid_utc must not contain missing timestamps")
    d[value_col] = d[value_col].where(np.isfinite(d[value_col]))
    timestamp_keys = ["issue_date", "point_id", *group_extra, "valid_utc"]
    for column in timestamp_keys:
        if d[column].isna().any():
            raise ValueError(f"{column} must not contain missing values")
    d = d.groupby(timestamp_keys, observed=True, as_index=False)[value_col].mean()
    d = add_local_date(d, offsets, time_col="valid_utc")
    d["lead"] = (pd.to_datetime(d["local_date"]) - pd.to_datetime(d["issue_date"])).dt.days
    keys = ["issue_date", "point_id", *group_extra, "lead"]
    g = d.groupby(keys, observed=True)[value_col].agg(
        mean="mean", max="max", size="count").reset_index()
    g.loc[g["size"] < min_count, ["mean", "max"]] = np.nan
    return g
