"""Tests for analyst/analysis.py — RFM, cohorts, basket, elasticity, matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst import analysis as A


def _orders_df(n_customers=50, days=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01")
    products = ["P1", "P2", "P3", "P4"]
    for i in range(n_customers):
        # Each customer makes 1..10 purchases at random points
        n_orders = int(rng.integers(1, 10))
        days_off = rng.integers(0, days, n_orders)
        for d in days_off:
            rows.append({
                "customer_id": f"C{i:03d}",
                "order_date": start + pd.Timedelta(days=int(d)),
                "product_id": products[int(rng.integers(0, len(products)))],
                "quantity": int(rng.integers(1, 5)),
                "revenue": float(rng.uniform(10, 200)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# RFM
# --------------------------------------------------------------------------- #

def test_rfm_returns_segments():
    df = _orders_df()
    rmap = {"customer": "customer_id", "date": "order_date", "amount": "revenue"}
    out = A.rfm(df, rmap)
    assert not out.empty
    assert {"customer", "R", "F", "M", "rfm_score", "segment"}.issubset(out.columns)
    assert out["segment"].isin(
        {"Champions", "Loyal", "New", "At-Risk", "Big Spenders Lost", "Lost", "Average"}
    ).all()


def test_rfm_handles_missing_role():
    out = A.rfm(pd.DataFrame({"x": [1]}), {"customer": "x"})  # no date/amount
    assert out.empty


# --------------------------------------------------------------------------- #
# Cohort retention
# --------------------------------------------------------------------------- #

def test_cohort_retention_first_column_is_one():
    df = _orders_df()
    rmap = {"customer": "customer_id", "date": "order_date"}
    mat = A.cohort_retention(df, rmap)
    assert not mat.empty
    # Month-0 retention is by definition 1.0 for any active cohort
    assert (mat.iloc[:, 0].dropna() == 1.0).all()


# --------------------------------------------------------------------------- #
# Market basket
# --------------------------------------------------------------------------- #

def test_market_basket_finds_pairs():
    # Construct baskets with strong P1+P2 co-purchase
    rows = []
    for i in range(20):
        rows.append({"basket": f"B{i}", "product_id": "P1"})
        rows.append({"basket": f"B{i}", "product_id": "P2"})
    for i in range(10):
        rows.append({"basket": f"B{20+i}", "product_id": "P3"})
    df = pd.DataFrame(rows)
    rmap = {"product": "product_id"}
    out = A.market_basket(df, rmap, basket_key="basket")
    assert not out.empty
    top = out.iloc[0]
    assert {top["item_a"], top["item_b"]} == {"P1", "P2"}
    assert top["lift"] > 1.0


def test_market_basket_handles_no_basket_key():
    df = pd.DataFrame({"product_id": ["P1", "P2"]})
    out = A.market_basket(df, {"product": "product_id"})
    assert out.empty  # no basket signal


# --------------------------------------------------------------------------- #
# Elasticity
# --------------------------------------------------------------------------- #

def test_elasticity_detects_negative_slope():
    # Synthetic: q = 10000 * p^-1.5 → log-log slope ≈ -1.5
    # Higher coefficient keeps q well above the int(...)/floor=1 region.
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(120):
        p = float(rng.uniform(5, 50))
        q = max(1, int(10000 * p ** -1.5 * rng.uniform(0.95, 1.05)))
        rows.append({"product_id": "P1", "price": p, "quantity": q})
    df = pd.DataFrame(rows)
    rmap = {"product": "product_id", "quantity": "quantity", "price": "price"}
    out = A.elasticity(df, rmap, price_col="price")
    assert not out.empty
    # With this much data + tight noise, slope should land within ±0.4 of -1.5
    assert -2.0 < out.iloc[0]["elasticity"] < -1.0
    assert out.iloc[0]["r2"] > 0.8


def test_elasticity_skips_thin_products():
    df = pd.DataFrame({"product_id": ["P1"] * 3, "price": [1.0, 2.0, 3.0],
                       "quantity": [10, 20, 30]})
    out = A.elasticity(df, {"product": "product_id", "quantity": "quantity"},
                       price_col="price")
    assert out.empty  # < 10 rows


# --------------------------------------------------------------------------- #
# Product matrix
# --------------------------------------------------------------------------- #

def test_product_matrix_assigns_quadrants():
    df = _orders_df()
    rmap = {"product": "product_id", "amount": "revenue", "date": "order_date"}
    out = A.product_matrix(df, rmap)
    assert not out.empty
    assert {"product", "revenue", "share", "growth", "quadrant"}.issubset(out.columns)
    assert set(out["quadrant"]).issubset({"Star", "Cash Cow", "Question Mark", "Dog"})


# --------------------------------------------------------------------------- #
# Revenue trend
# --------------------------------------------------------------------------- #

def test_revenue_trend_weekly():
    df = _orders_df()
    rmap = {"date": "order_date", "amount": "revenue"}
    out = A.revenue_trend(df, rmap, freq="W")
    assert not out.empty
    assert {"period", "revenue", "pct_change", "rolling_4"}.issubset(out.columns)
