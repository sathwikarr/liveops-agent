"""Simple revenue forecasting utilities used by the Dashboard.

This module provides a single public function `forecast_revenue` which
accepts either a path-like to a CSV or a pandas.DataFrame and returns a
DataFrame compatible with the dashboard: columns `ds`, `yhat`, `yhat_lower`,
`yhat_upper`, and `anomaly` (boolean).

Implementation uses Prophet (installed in requirements) when enough
historical data is available; otherwise returns an empty DataFrame.
"""
from __future__ import annotations

from typing import Union
import pandas as pd
import numpy as np
from pathlib import Path

try:
    # prophet package (PyPI) provides Prophet
    from prophet import Prophet
except Exception:
    Prophet = None


def _read_input(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    p = Path(data)
    return pd.read_csv(p, on_bad_lines="skip")


def forecast_revenue(data: Union[str, Path, pd.DataFrame], periods: int = 10) -> pd.DataFrame:
    """Return a forecast DataFrame usable by the dashboard.

    Expects a time column named `timestamp` (or `date`) and a numeric
    `revenue` column. Aggregates daily revenue before modeling.

    If Prophet is unavailable or there is insufficient history (< 10 days),
    an empty DataFrame is returned.
    """
    try:
        df = _read_input(data)
    except Exception:
        return pd.DataFrame()

    # Normalize column names
    df_cols = {c.lower(): c for c in df.columns}
    time_col = df_cols.get("timestamp") or df_cols.get("date")
    rev_col = df_cols.get("revenue")

    if time_col is None or rev_col is None:
        return pd.DataFrame()

    # Prepare series: daily sum
    try:
        df["_ts"] = pd.to_datetime(df[time_col], errors="coerce")
    except Exception:
        return pd.DataFrame()

    df = df.dropna(subset=["_ts"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["_rev"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0.0)
    daily = df.set_index("_ts").resample("D")["_rev"].sum().reset_index()
    daily.rename(columns={"_ts": "ds", "_rev": "y"}, inplace=True)

    if len(daily) < 10 or Prophet is None:
        # Not enough history or prophet not available — return empty
        return pd.DataFrame()

    try:
        m = Prophet()
        m.fit(daily)
        future = m.make_future_dataframe(periods=periods)
        fcst = m.predict(future)

        # Keep only the columns we need
        out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        # No observed 'y' in the forecasted future periods — mark anomaly False
        out["anomaly"] = False

        # For convenience, where historical observed values exist, flag if observed outside intervals
        hist = daily.set_index("ds")
        out = out.set_index("ds")
        common = out.index.intersection(hist.index)
        if not common.empty:
            obs = hist.loc[common, "y"]
            out.loc[common, "anomaly"] = (
                (obs < out.loc[common, "yhat_lower"]) | (obs > out.loc[common, "yhat_upper"])
            ).values

        out = out.reset_index()
        return out
    except Exception:
        return pd.DataFrame()


__all__ = ["forecast_revenue"]
