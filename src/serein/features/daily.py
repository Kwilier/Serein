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
    off = df[id_col].map(offsets)
    local = df[time_col] + pd.to_timedelta(off, unit="h")
    df = df.assign(local_date=local.dt.date, _off=off)
    return df


def obs_daily_means(hourly: pd.DataFrame, offsets: pd.Series,
                    min_hours: int = MIN_HOURS_DAILY) -> pd.DataFrame:
    """Daily mean PM2.5; require min_hours distinct finite hours per local day.

    Repeated site/timestamp readings are averaged before daily aggregation so
    each hour has equal weight and counts only once toward completeness.
    """
    h = hourly.rename(columns={"site_id": "point_id"})
    h = h[h["point_id"].isin(offsets.index)]
    h["pm25"] = h["pm25"].where(np.isfinite(h["pm25"]))
    h = h.groupby(["point_id", "time_utc"], as_index=False)["pm25"].mean()
    h = add_local_date(h, offsets)
    g = h.groupby(["point_id", "local_date"])["pm25"].agg(
        mean="mean", size="count").reset_index()
    g.loc[g["size"] < min_hours, "mean"] = np.nan
    return g.rename(columns={"mean": "pm25_daily", "size": "n_hours"})


def model_daily(df: pd.DataFrame, offsets: pd.Series, value_col: str = "value",
                min_count: int = 12, group_extra: tuple = ()) -> pd.DataFrame:
    """Aggregate a forecast (issue_date, point_id, valid_utc, value) to lead-day mean/max.

    lead = local_date - issue_date (days). Size counts finite values.
    """
    d = add_local_date(df, offsets, time_col="valid_utc")
    d[value_col] = d[value_col].where(np.isfinite(d[value_col]))
    d["lead"] = (pd.to_datetime(d["local_date"]) - pd.to_datetime(d["issue_date"])).dt.days
    keys = ["issue_date", "point_id", *group_extra, "lead"]
    g = d.groupby(keys, observed=True)[value_col].agg(
        mean="mean", max="max", size="count").reset_index()
    g.loc[g["size"] < min_count, ["mean", "max"]] = np.nan
    return g
