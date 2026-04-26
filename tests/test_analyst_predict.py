"""Tests for analyst/predict.py — churn, stockout, demand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst import predict as P


def test_churn_recent_customer_low_score():
    today = pd.Timestamp("2025-04-01")
    df = pd.DataFrame({
        "customer_id": ["C1"] * 5,
        "order_date": pd.date_range(today - pd.Timedelta(days=4), today),
    })
    out = P.churn_scores(df, {"customer": "customer_id", "date": "order_date"})
    assert not out.empty
    assert out.iloc[0]["churn_prob"] < 0.2
    assert out.iloc[0]["risk"] == "Active"


def test_churn_old_customer_high_score():
    df = pd.DataFrame({
        "customer_id": ["A", "A", "B"],
        "order_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"),
                       pd.Timestamp("2025-04-01")],
    })
    out = P.churn_scores(df, {"customer": "customer_id", "date": "order_date"})
    a = out[out["customer"] == "A"].iloc[0]
    assert a["churn_prob"] > 0.8
    assert a["risk"] == "Churned"


def test_churn_handles_missing_role():
    out = P.churn_scores(pd.DataFrame({"x": [1]}), {"date": "x"})  # no customer
    assert out.empty


def test_stockout_horizon_critical_when_velocity_high():
    today = pd.Timestamp("2025-04-01")
    rows = []
    # Daily sales of 10 units of P1 for 14 days, with on-hand of 30
    for i in range(14):
        rows.append({"product_id": "P1", "order_date": today - pd.Timedelta(days=i),
                     "quantity": 10, "inventory": 30})
    rows.append({"product_id": "P2", "order_date": today, "quantity": 1, "inventory": 1000})
    df = pd.DataFrame(rows)
    rmap = {"product": "product_id", "date": "order_date",
            "quantity": "quantity", "inventory": "inventory"}
    out = P.stockout_horizon(df, rmap)
    p1 = out[out["product"] == "P1"].iloc[0]
    assert p1["risk"] == "Critical"
    assert 0 < p1["days_to_stockout"] <= 7


def test_stockout_handles_no_inventory_column():
    df = pd.DataFrame({"product_id": ["P1"], "order_date": [pd.Timestamp("2025-01-01")],
                       "quantity": [5]})
    out = P.stockout_horizon(df, {"product": "product_id", "date": "order_date",
                                  "quantity": "quantity"})
    # on_hand will be NaN, days_to_stockout NaN, risk Unknown — but should not crash
    if not out.empty:
        assert out.iloc[0]["risk"] == "Unknown"


def test_demand_per_segment_returns_dict():
    """Smoke: function returns a dict (possibly empty if Prophet missing)."""
    df = pd.DataFrame({
        "order_date": pd.date_range("2025-01-01", periods=120, freq="D"),
        "region": ["E"] * 120,
        "product_id": ["P1"] * 120,
        "revenue": np.linspace(100, 200, 120),
        "quantity": [1] * 120,
    })
    rmap = {"date": "order_date", "region": "region",
            "product": "product_id", "amount": "revenue", "quantity": "quantity"}
    out = P.demand_per_segment(df, rmap, periods=5, top_n=2)
    assert isinstance(out, dict)
