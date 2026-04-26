"""Tests for agent/forecast.py — segments, backtest, helpers.

These rely on Prophet being installed. If Prophet is missing, the public
functions return empty DataFrames cleanly — we check that path too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import agent.forecast as F


def _synthetic(n_days=120, segments=(("East", "P1"), ("West", "P2")), seed=42):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for region, pid in segments:
        base = rng.uniform(80, 200)
        trend = np.linspace(0, 30, n_days)
        season = 20 * np.sin(np.arange(n_days) / 7 * 2 * np.pi)
        noise = rng.normal(0, 8, n_days)
        rev = base + trend + season + noise
        for d, r in zip(days, rev):
            rows.append({
                "timestamp": d,
                "region": region, "product_id": pid,
                "orders": int(max(1, r / 12)),
                "inventory": 100,
                "revenue": float(max(0, r)),
            })
    return pd.DataFrame(rows)


def test_recent_top_segments_returns_pairs():
    df = _synthetic()
    top = F.recent_top_segments(df, 2)
    assert isinstance(top, list)
    assert len(top) == 2
    assert all(isinstance(x, tuple) and len(x) == 2 for x in top)


def test_recent_top_segments_handles_missing_cols():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    assert F.recent_top_segments(df, 5) == []


def test_recent_top_segments_handles_empty():
    assert F.recent_top_segments(pd.DataFrame(), 5) == []


def test_normalize_handles_missing_cols():
    df = pd.DataFrame({"foo": [1]})
    assert F._normalize(df) is None


def test_infer_freq_daily():
    daily = pd.DataFrame({
        "ds": pd.date_range("2025-01-01", periods=5, freq="D"),
        "y": [1, 2, 3, 4, 5],
    })
    assert F._infer_freq(daily) == "D"


def test_infer_freq_hourly():
    hourly = pd.DataFrame({
        "ds": pd.date_range("2025-01-01", periods=5, freq="h"),
        "y": [1, 2, 3, 4, 5],
    })
    assert F._infer_freq(hourly) == "h"


@pytest.mark.skipif(not F.HAS_PROPHET, reason="Prophet not installed")
def test_forecast_segment_returns_nonneg_yhat():
    df = _synthetic()
    out = F.forecast_segment(df, "East", "P1", periods=10)
    assert not out.empty
    assert (out["yhat"] >= 0).all()
    assert (out["yhat_lower"] >= 0).all()
    for col in ("ds", "yhat", "yhat_lower", "yhat_upper"):
        assert col in out.columns


@pytest.mark.skipif(not F.HAS_PROPHET, reason="Prophet not installed")
def test_forecast_revenue_global():
    df = _synthetic()
    out = F.forecast_revenue(df, periods=5)
    assert not out.empty
    assert (out["yhat"] >= 0).all()


@pytest.mark.skipif(not F.HAS_PROPHET, reason="Prophet not installed")
def test_forecast_per_segment_returns_dict():
    df = _synthetic()
    out = F.forecast_per_segment(df, periods=5, top_n=2)
    assert isinstance(out, dict)
    assert len(out) >= 1
    for k, v in out.items():
        assert isinstance(k, tuple) and len(k) == 2
        assert isinstance(v, pd.DataFrame)
        assert not v.empty


@pytest.mark.skipif(not F.HAS_PROPHET, reason="Prophet not installed")
def test_backtest_segment_low_mape_on_synthetic():
    df = _synthetic()
    bt = F.backtest_segment(df, "East", "P1", horizon=5, folds=3)
    assert bt["n_folds"] >= 1
    # Synthetic data is well-fit by Prophet — MAPE should be small.
    assert bt["mape"] < 25.0


def test_backtest_segment_handles_short_history():
    # Only 1 day: span=0, _pick_freq returns "15min", but we only have 1 daily
    # synthetic point — way below min_train + horizon buckets.
    df = _synthetic(n_days=1)
    bt = F.backtest_segment(df, "East", "P1", horizon=3, folds=3, min_train=14)
    assert bt["n_folds"] == 0


def test_no_prophet_returns_empty_frames(monkeypatch):
    """Without Prophet, public functions degrade to empty frames."""
    monkeypatch.setattr(F, "HAS_PROPHET", False)
    df = _synthetic(n_days=30)
    assert F.forecast_revenue(df, 5).empty
    assert F.forecast_segment(df, "East", "P1", 5).empty
    bt = F.backtest_segment(df, "East", "P1", horizon=3, folds=2)
    assert bt["n_folds"] == 0
