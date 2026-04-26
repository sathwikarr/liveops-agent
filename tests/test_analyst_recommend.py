"""Tests for analyst/recommend.py — recommendation generators."""
from __future__ import annotations

import pandas as pd
import pytest

from analyst import recommend as R


def test_from_rfm_emits_winback_for_at_risk():
    df = pd.DataFrame({
        "customer": [f"C{i}" for i in range(10)],
        "monetary": [100.0] * 10,
        "segment": ["At-Risk"] * 10,
        "R": [1] * 10, "F": [3] * 10, "M": [3] * 10,
    })
    recs = R.from_rfm(df)
    assert any("win-back" in r.action.lower() for r in recs)


def test_from_elasticity_recommends_price_cut():
    df = pd.DataFrame({
        "product": ["P1"], "n": [50], "mean_price": [10.0],
        "mean_qty": [25.0], "elasticity": [-1.5], "r2": [0.85],
    })
    recs = R.from_elasticity(df)
    assert any("drop" in r.action.lower() and "price" in r.action.lower() for r in recs)
    assert recs[0].category == "pricing"


def test_from_elasticity_recommends_price_raise_when_inelastic():
    df = pd.DataFrame({
        "product": ["P1"], "n": [50], "mean_price": [10.0],
        "mean_qty": [25.0], "elasticity": [-0.1], "r2": [0.7],
    })
    recs = R.from_elasticity(df)
    assert any("price increase" in r.action.lower() for r in recs)


def test_from_elasticity_skips_low_r2():
    df = pd.DataFrame({
        "product": ["P1"], "n": [50], "mean_price": [10.0],
        "mean_qty": [25.0], "elasticity": [-2.0], "r2": [0.1],
    })
    assert R.from_elasticity(df) == []


def test_from_product_matrix_boosts_stars():
    df = pd.DataFrame({
        "product": ["P1", "P2", "P3", "P4"],
        "revenue": [1000, 800, 50, 40],
        "share": [0.5, 0.4, 0.05, 0.05],
        "growth": [0.3, 0.2, -0.1, -0.2],
        "quadrant": ["Star", "Star", "Dog", "Dog"],
    })
    recs = R.from_product_matrix(df)
    assert any("ad spend" in r.action.lower() for r in recs)


def test_from_basket_emits_bundle():
    df = pd.DataFrame({
        "item_a": ["P1"], "item_b": ["P2"],
        "support": [0.3], "confidence": [0.7],
        "lift": [3.5], "n_baskets": [50],
    })
    recs = R.from_basket(df)
    assert any("bundle" in r.action.lower() or "cross-sell" in r.action.lower() for r in recs)


def test_from_stockout_critical_emits_reorder():
    df = pd.DataFrame({
        "product": ["P1"], "on_hand": [30.0], "daily_demand": [10.0],
        "days_to_stockout": [3.0], "risk": ["Critical"],
    })
    recs = R.from_stockout(df)
    assert any("reorder" in r.action.lower() for r in recs)
    assert recs[0].confidence == "High"


def test_from_churn_atrisk_emits_offer():
    df = pd.DataFrame({
        "customer": [f"C{i}" for i in range(10)],
        "days_since": [40] * 10,
        "churn_prob": [0.6] * 10,
        "risk": ["At-Risk"] * 10,
    })
    recs = R.from_churn(df)
    assert any("offer" in r.action.lower() or "free-ship" in r.action.lower() for r in recs)


def test_generate_returns_sorted_by_score():
    rfm = pd.DataFrame({
        "customer": [f"C{i}" for i in range(10)],
        "monetary": [100.0] * 10,
        "segment": ["At-Risk"] * 10,
        "R": [1] * 10, "F": [3] * 10, "M": [3] * 10,
    })
    stock = pd.DataFrame({
        "product": ["P1"], "on_hand": [30.0], "daily_demand": [10.0],
        "days_to_stockout": [3.0], "risk": ["Critical"],
    })
    recs = R.generate(rfm_df=rfm, stockout_df=stock)
    assert len(recs) >= 2
    assert all(recs[i].score >= recs[i + 1].score for i in range(len(recs) - 1))


def test_generate_with_no_inputs_returns_empty():
    assert R.generate() == []
