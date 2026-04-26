"""Tests for analyst/eda.py — distributions, correlations, outliers, seasonality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst import ingest as I
from analyst import eda as E


def _sales_df(n=120, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=n, freq="D")
    # Inject weekly seasonality: weekend ×2
    base = rng.uniform(80, 120, n)
    weekend_boost = np.where(days.dayofweek >= 5, 1.8, 1.0)
    rev = base * weekend_boost
    df = pd.DataFrame({
        "order_date": days,
        "region": np.tile(["East", "West"], n // 2),
        "revenue": rev,
        "quantity": rng.integers(1, 10, n),
    })
    return df


def test_profile_returns_numeric_and_dates():
    df = _sales_df()
    schema = I.infer_schema(df.astype({"revenue": str, "quantity": str}))
    # Use the schema's coerced view for stats
    df2 = schema.coerced
    df2["order_date"] = pd.to_datetime(df2["order_date"])
    report = E.profile(df2, schema)
    assert report.n_rows == len(df)
    assert any(d.name == "revenue" for d in report.numeric)
    assert report.dates and report.dates[0].name == "order_date"


def test_correlations_detect_strong_pair():
    df = pd.DataFrame({
        "x": np.arange(100, dtype=float),
        "y": np.arange(100, dtype=float) * 2 + 5,
        "z": np.random.default_rng(1).normal(0, 1, 100),
    })
    schema = I.infer_schema(df.astype(str))
    coerced = schema.coerced
    report = E.profile(coerced, schema)
    pair = report.correlations[0]
    assert {pair.a, pair.b} == {"x", "y"}
    assert pair.pearson > 0.99


def test_outliers_detected():
    rng = np.random.default_rng(2)
    vals = rng.normal(100, 5, 200).tolist()
    vals[0] = 10000  # huge outlier
    df = pd.DataFrame({"x": vals})
    schema = I.infer_schema(df.astype(str))
    report = E.profile(schema.coerced, schema)
    assert any(o.column == "x" and o.n_outliers > 0 for o in report.outliers)


def test_seasonality_weekend_peak():
    df = _sales_df()
    schema = I.infer_schema(df.astype({"revenue": str, "quantity": str}))
    df2 = schema.coerced
    df2["order_date"] = pd.to_datetime(df2["order_date"])
    role_map = {"date": "order_date", "amount": "revenue"}
    report = E.profile(df2, schema, kind="sales", role_map=role_map)
    assert report.seasonality is not None
    assert report.seasonality.has_weekly is True
    assert report.seasonality.peak_weekday in {"Saturday", "Sunday"}


def test_seasonality_handles_missing_role_map():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    schema = I.infer_schema(df.astype(str))
    report = E.profile(schema.coerced, schema)
    assert report.seasonality is not None
    assert "Need both" in (report.seasonality.notes[0] if report.seasonality.notes else "")


def test_headline_lines_are_present():
    df = _sales_df()
    schema = I.infer_schema(df.astype({"revenue": str, "quantity": str}))
    df2 = schema.coerced
    df2["order_date"] = pd.to_datetime(df2["order_date"])
    report = E.profile(df2, schema, kind="sales",
                       role_map={"date": "order_date", "amount": "revenue"})
    assert report.headline
    assert any("rows" in line for line in report.headline)


def test_narrate_falls_back_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    df = _sales_df()
    schema = I.infer_schema(df.astype({"revenue": str, "quantity": str}))
    df2 = schema.coerced
    df2["order_date"] = pd.to_datetime(df2["order_date"])
    report = E.profile(df2, schema, kind="sales",
                       role_map={"date": "order_date", "amount": "revenue"})
    text = E.narrate(report, kind="sales")
    assert isinstance(text, str) and len(text) > 0
