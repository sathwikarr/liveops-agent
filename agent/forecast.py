# agent/forecast.py
import os
import pandas as pd
import numpy as np

MIN_ROWS = 30  # need at least this many points to fit a reasonable model

def _infer_freq(ds: pd.Series) -> str:
    """Infer a sensible frequency from timestamp deltas."""
    ds = pd.to_datetime(ds).sort_values().dropna()
    if len(ds) < 3:
        return "D"
    deltas = ds.diff().dropna().value_counts().index
    if not len(deltas):
        return "D"
    median_delta = deltas[0]
    minutes = median_delta.total_seconds() / 60.0
    if minutes <= 1.5:
        return "min"
    if minutes <= 30:
        return "15min"
    if minutes <= 90:
        return "H"
    if minutes <= 48 * 60:
        return "D"
    return "D"

def forecast_revenue(csv_file: str, periods: int = 10) -> pd.DataFrame:
    """
    Fit Prophet on historical revenue and forecast `periods` future steps.
    Returns df with ['ds','yhat','yhat_lower','yhat_upper','anomaly'].
    'anomaly' marks future timestamps where the lower bound is below a
    robust threshold (median - 1.5 * MAD) of historical revenue.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return pd.DataFrame()

    # Basic schema guard
    if not {"timestamp", "revenue"}.issubset(df.columns):
        return pd.DataFrame()

    # Clean & prepare
    df = df[["timestamp", "revenue"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df = df.dropna(subset=["timestamp", "revenue"])
    if df.empty or len(df) < MIN_ROWS:
        return pd.DataFrame()

    # Sort, de-dup timestamps (take mean per timestamp)
    df = df.sort_values("timestamp").groupby("timestamp", as_index=False)["revenue"].mean()

    # Prophet requires columns ds, y
    hist = df.rename(columns={"timestamp": "ds", "revenue": "y"})

    # Prophet does better with non-negative values; clip tiny negatives from noise
    hist["y"] = hist["y"].clip(lower=0)

    # Infer a reasonable frequency from the data
    freq = _infer_freq(hist["ds"])

    # Lazy import Prophet so the app doesn't crash if not installed
    try:
        from prophet import Prophet
    except Exception:
        # Prophet not installed or failed to import
        return pd.DataFrame()

    # Configure model: enable seasonality that matches cadence
    daily = freq in ("min", "15min", "H", "D")
    weekly = True
    yearly = len(hist) >= 180  # only if long history

    try:
        m = Prophet(
            daily_seasonality=daily,
            weekly_seasonality=weekly,
            yearly_seasonality=yearly,
            changepoint_prior_scale=0.25,  # a tad smoother
            interval_width=0.9,            # wider PI for safer anomaly flags
        )
        m.fit(hist)
    except Exception:
        return pd.DataFrame()

    # Build future frame using inferred frequency
    try:
        future = m.make_future_dataframe(periods=periods, freq=freq, include_history=False)
        fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception:
        return pd.DataFrame()

    # Robust threshold from historical revenue (median ± 1.5 * MAD)
    y = hist["y"].values
    median = np.median(y)
    mad = np.median(np.abs(y - median)) or 1e-9
    lower_thresh = median - 1.5 * mad

    # Flag future points whose lower bound dips below threshold
    fcst["anomaly"] = fcst["yhat_lower"] < lower_thresh
    return fcst
