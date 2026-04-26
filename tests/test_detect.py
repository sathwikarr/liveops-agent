"""Tests for agent/detect.py — z-score and IsolationForest paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent.detect import (
    DEFAULT_METRICS,
    detect_anomalies,
    zscore_anomaly_detection,
    _zscore_series,
)


def _build_df(n=120, inject_spike=True, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for region in ("East", "West"):
        for pid in ("P1", "P2"):
            base = rng.uniform(80, 200)
            for i in range(n):
                rev = base + rng.normal(0, 8)
                rows.append({
                    "timestamp": f"2025-01-01T{i % 24:02d}:00:00",
                    "region": region, "product_id": pid,
                    "orders": int(max(1, rev / 12)),
                    "inventory": 100,
                    "revenue": float(max(0, rev)),
                })
    df = pd.DataFrame(rows)
    if inject_spike:
        # Inject a clear spike in (East, P1) — 10x the normal level.
        df.loc[(df["region"] == "East") & (df["product_id"] == "P1"), "revenue"].iloc[0]
        df.iloc[0, df.columns.get_loc("revenue")] = 5000.0
    return df


def test_zscore_series_basic():
    s = pd.Series([1.0, 1.0, 1.0, 1.0, 5.0])
    z = _zscore_series(s)
    assert not np.isnan(z.iloc[-1])
    # The 5.0 is clearly the outlier in this distribution.
    assert abs(z.iloc[-1]) > abs(z.iloc[0])


def test_zscore_series_handles_constant():
    s = pd.Series([3.0] * 10)
    z = _zscore_series(s)
    assert z.isna().all()  # std == 0 → all NaN


def test_zscore_series_handles_short_sample():
    s = pd.Series([1.0, 2.0])  # < 3 datapoints
    z = _zscore_series(s)
    assert z.isna().all()


def test_detect_anomalies_empty_returns_empty_frame():
    df = pd.DataFrame()
    out = detect_anomalies(df)
    assert out.empty
    assert "metric" in out.columns


def test_detect_anomalies_finds_injected_spike():
    df = _build_df()
    out = detect_anomalies(df, threshold=2.0, method="zscore")
    assert not out.empty
    # The injected 5000 spike should appear at the top.
    top = out.iloc[0]
    assert top["region"] == "East"
    assert top["product_id"] == "P1"
    assert top["metric"] == "revenue"
    assert abs(top["z_score"]) > 2.0


def test_detect_anomalies_returns_required_columns():
    df = _build_df()
    out = detect_anomalies(df, threshold=2.0)
    expected = {"timestamp", "region", "product_id", "metric", "value",
                "z_score", "scope", "orders", "inventory", "revenue"}
    assert expected.issubset(set(out.columns))


def test_detect_anomalies_method_options_all_run():
    df = _build_df()
    for method in ("zscore", "auto", "isoforest"):
        out = detect_anomalies(df, threshold=2.0, method=method)
        # All three should run without error and produce a DataFrame.
        assert isinstance(out, pd.DataFrame)


def test_detect_anomalies_threshold_filters():
    df = _build_df()
    permissive = detect_anomalies(df, threshold=1.0, method="zscore")
    strict = detect_anomalies(df, threshold=3.5, method="zscore")
    assert len(permissive) >= len(strict)


def test_legacy_zscore_anomaly_detection():
    df = _build_df()
    out = zscore_anomaly_detection(df, "revenue", threshold=2.0)
    assert isinstance(out, pd.DataFrame)
    assert "z_score" in out.columns


def test_default_metrics_sane():
    assert "revenue" in DEFAULT_METRICS
    assert "orders" in DEFAULT_METRICS
    assert "inventory" in DEFAULT_METRICS
