"""Revenue forecasting — global + per-segment + backtests.

Public API:
- `forecast_revenue(data, periods)`: global daily revenue forecast (back-compat).
- `forecast_per_segment(df, periods, top_n)`: dict of forecasts keyed by
  (region, product_id), one Prophet fit per segment with enough history.
- `backtest_segment(df, region, product_id, horizon, folds)`: walk-forward
  CV returning MAPE / SMAPE / RMSE for a single segment.
- `recent_top_segments(df, n)`: helper for the dashboard to pick which
  segments to surface.

All Prophet calls are wrapped in try/except — Prophet is heavy and not always
installed in test envs, so we degrade to empty DataFrames cleanly.

Forecasts are cached in-process keyed on (segment, df-content-hash, periods).
Streamlit reruns on the same uploaded CSV are effectively free.
"""
from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

try:
    from prophet import Prophet  # type: ignore
    HAS_PROPHET = True
except Exception:
    Prophet = None  # type: ignore
    HAS_PROPHET = False

# Silence Prophet/cmdstanpy chatter
for noisy in ("cmdstanpy", "prophet", "prophet.forecaster"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def _read_input(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(Path(data), on_bad_lines="skip")


def _normalize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Find timestamp + revenue columns regardless of case; return parsed copy."""
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("timestamp") or cols.get("date") or cols.get("ds")
    rev_col = cols.get("revenue") or cols.get("y")
    if not time_col or not rev_col:
        return None
    out = df.copy()
    out["_ts"] = pd.to_datetime(out[time_col], errors="coerce")
    out["_rev"] = pd.to_numeric(out[rev_col], errors="coerce")
    out = out.dropna(subset=["_ts", "_rev"])
    return out if not out.empty else None


def _pick_freq(df: pd.DataFrame) -> str:
    """Pick a resample frequency that yields enough Prophet points.

    Real demo CSVs are often minute-level over a single day. Falling straight
    to D would collapse them to 1-2 rows. We escalate granularity until the
    series has at least ~20 buckets.
    """
    ts = df["_ts"]
    span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
    if span_days >= 25:
        return "D"
    if span_days >= 3:
        return "h"
    return "15min"


def _daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Prophet-ready DataFrame with columns ds (date), y (revenue).

    Despite the name, the resample frequency is chosen adaptively based on the
    span of the series so short intraday CSVs still fit.
    """
    freq = _pick_freq(df)
    daily = (
        df.set_index("_ts").resample(freq)["_rev"].sum().reset_index()
        .rename(columns={"_ts": "ds", "_rev": "y"})
    )
    return daily


def _hash_df(df: pd.DataFrame) -> str:
    """Stable hash of the rows we care about, for cache keys."""
    cols = [c for c in ["timestamp", "region", "product_id", "revenue"] if c in df.columns]
    if not cols:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
    sub = df[cols].astype(str)
    return hashlib.md5(pd.util.hash_pandas_object(sub, index=False).values.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Core fit + predict
# --------------------------------------------------------------------------- #

def _infer_freq(daily: pd.DataFrame) -> str:
    """Infer a pandas offset alias from the median spacing of the series."""
    if len(daily) < 2:
        return "D"
    deltas = daily["ds"].diff().dropna()
    if deltas.empty:
        return "D"
    secs = deltas.dt.total_seconds().median()
    if secs <= 60:
        return "min"
    if secs <= 60 * 15:
        return "15min"
    if secs <= 60 * 60:
        return "h"
    return "D"


def _fit_predict(daily: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Fit Prophet, return forecast DataFrame with ds/yhat/yhat_lower/yhat_upper/anomaly."""
    if not HAS_PROPHET or len(daily) < 10:
        return pd.DataFrame()

    freq = _infer_freq(daily)
    try:
        m = Prophet()
        m.fit(daily)
        future = m.make_future_dataframe(periods=periods, freq=freq)
        fcst = m.predict(future)
    except Exception:
        return pd.DataFrame()

    out = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    # Revenue can't be negative — clip the forecast and bands at zero.
    for c in ("yhat", "yhat_lower", "yhat_upper"):
        out[c] = out[c].clip(lower=0.0)
    out["anomaly"] = False
    hist = daily.set_index("ds")
    out_idx = out.set_index("ds")
    common = out_idx.index.intersection(hist.index)
    if not common.empty:
        obs = hist.loc[common, "y"]
        out_idx.loc[common, "anomaly"] = (
            (obs < out_idx.loc[common, "yhat_lower"]) | (obs > out_idx.loc[common, "yhat_upper"])
        ).values
        out_idx.loc[common, "y_actual"] = obs.values
    return out_idx.reset_index()


# --------------------------------------------------------------------------- #
# Public: global forecast (back-compat)
# --------------------------------------------------------------------------- #

def forecast_revenue(data: Union[str, Path, pd.DataFrame], periods: int = 10) -> pd.DataFrame:
    """Global daily-revenue forecast. Kept for back-compat with old dashboards."""
    try:
        df = _read_input(data)
    except Exception:
        return pd.DataFrame()
    n = _normalize(df)
    if n is None:
        return pd.DataFrame()
    return _fit_predict(_daily_series(n), periods)


# --------------------------------------------------------------------------- #
# Public: segments
# --------------------------------------------------------------------------- #

def recent_top_segments(df: pd.DataFrame, n: int = 6) -> list[tuple[str, str]]:
    """Pick top n (region, product_id) by recent revenue. Stable, deterministic."""
    if df is None or df.empty:
        return []
    needed = {"region", "product_id", "revenue"}
    if not needed.issubset(df.columns):
        return []
    rev = (
        df.assign(revenue=pd.to_numeric(df["revenue"], errors="coerce"))
          .groupby(["region", "product_id"], dropna=False)["revenue"]
          .sum().sort_values(ascending=False).head(n)
    )
    return [(str(r), str(p)) for (r, p), _ in rev.items()]


def _segment_daily(df: pd.DataFrame, region: str, product_id: str) -> Optional[pd.DataFrame]:
    sub = df[(df["region"].astype(str) == region) & (df["product_id"].astype(str) == product_id)]
    n = _normalize(sub)
    return _daily_series(n) if n is not None else None


# Cache keyed on segment + data hash + periods, scoped to a process.
@lru_cache(maxsize=128)
def _cached_segment_forecast(region: str, product_id: str, df_hash: str, periods: int,
                             daily_payload: tuple) -> tuple:
    """daily_payload is the daily series serialized to a hashable tuple."""
    daily = pd.DataFrame(list(daily_payload), columns=["ds", "y"])
    daily["ds"] = pd.to_datetime(daily["ds"])
    out = _fit_predict(daily, periods)
    return tuple(out.itertuples(index=False, name=None)), tuple(out.columns)


def forecast_segment(df: pd.DataFrame, region: str, product_id: str, periods: int = 10) -> pd.DataFrame:
    """Forecast a single (region, product_id) segment. Cached."""
    daily = _segment_daily(df, region, product_id)
    if daily is None or len(daily) < 10:
        return pd.DataFrame()
    payload = tuple((d.isoformat(), float(y)) for d, y in zip(daily["ds"], daily["y"]))
    rows, cols = _cached_segment_forecast(region, product_id, _hash_df(df), int(periods), payload)
    return pd.DataFrame(list(rows), columns=list(cols))


def forecast_per_segment(
    df: pd.DataFrame,
    periods: int = 10,
    top_n: int = 6,
    segments: Optional[Iterable[tuple[str, str]]] = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Forecast multiple segments. Defaults to the top_n by recent revenue."""
    targets = list(segments) if segments else recent_top_segments(df, top_n)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for region, product_id in targets:
        fc = forecast_segment(df, region, product_id, periods)
        if not fc.empty:
            out[(region, product_id)] = fc
    return out


# --------------------------------------------------------------------------- #
# Backtests
# --------------------------------------------------------------------------- #

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else float("nan")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = (denom != 0) & np.isfinite(denom)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


def backtest_segment(
    df: pd.DataFrame,
    region: str,
    product_id: str,
    horizon: int = 3,
    folds: int = 3,
    min_train: int = 14,
) -> dict:
    """Walk-forward backtest. Returns {'mape': %, 'smape': %, 'rmse': $, 'n_folds': k}."""
    daily = _segment_daily(df, region, product_id)
    if daily is None or len(daily) < min_train + horizon:
        return {"mape": float("nan"), "smape": float("nan"), "rmse": float("nan"), "n_folds": 0}

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    n = len(daily)
    freq = _infer_freq(daily)
    # Fold k uses the last (k+1)*horizon points as the held-out tail.
    completed = 0
    for k in range(folds):
        cutoff = n - (k + 1) * horizon
        if cutoff < min_train:
            break
        train = daily.iloc[:cutoff].copy()
        test = daily.iloc[cutoff:cutoff + horizon].copy()
        if not HAS_PROPHET or len(train) < 10 or test.empty:
            continue
        try:
            m = Prophet()
            m.fit(train)
            future = m.make_future_dataframe(periods=horizon, freq=freq)
            fcst = m.predict(future).set_index("ds")
            # Only keep test rows whose ds actually appears in the forecast index
            common = test["ds"][test["ds"].isin(fcst.index)]
            if common.empty:
                continue
            preds = np.clip(fcst.loc[common, "yhat"].values, 0.0, None)
            test = test.set_index("ds").loc[common]
        except Exception:
            continue
        # Skip folds where actuals are entirely zero — SMAPE/MAPE undefined.
        actuals = test["y"].astype(float).to_numpy()
        if not np.any(actuals != 0):
            continue
        y_true_all.extend(actuals.tolist())
        y_pred_all.extend(map(float, preds))
        completed += 1

    if completed == 0:
        return {"mape": float("nan"), "smape": float("nan"), "rmse": float("nan"), "n_folds": 0}

    y_true = np.array(y_true_all, dtype=float)
    y_pred = np.array(y_pred_all, dtype=float)
    return {
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "n_folds": completed,
    }


__all__ = [
    "forecast_revenue",
    "forecast_segment",
    "forecast_per_segment",
    "recent_top_segments",
    "backtest_segment",
]
