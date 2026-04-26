"""Tests for Stage 7 — NL query, what-if, narrative report."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from analyst import nlq, whatif, report as R


# --------------------------------------------------------------------------- #
# NL query
# --------------------------------------------------------------------------- #

def _df():
    return pd.DataFrame({
        "order_date": pd.date_range("2025-01-01", periods=20, freq="D"),
        "product_id": ["P1"] * 10 + ["P2"] * 10,
        "customer_id": [f"C{i}" for i in range(20)],
        "revenue": [100.0] * 10 + [50.0] * 10,
    })


def _rmap():
    return {"date": "order_date", "product": "product_id",
            "customer": "customer_id", "amount": "revenue"}


def test_nlq_total_revenue():
    out = nlq.ask("What is the total revenue?", _df(), _rmap())
    assert "1,500" in out.answer or "1500" in out.answer


def test_nlq_top_product():
    out = nlq.ask("Top 1 product", _df(), _rmap())
    assert "P1" in out.answer


def test_nlq_monthly_trend():
    out = nlq.ask("Show me revenue by month", _df(), _rmap())
    assert out.data is not None and len(out.data) >= 1


def test_nlq_unknown_falls_back_when_no_llm(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = nlq.ask("explain the meaning of life", _df(), _rmap())
    assert "couldn't" in out.answer.lower()


# --------------------------------------------------------------------------- #
# What-if
# --------------------------------------------------------------------------- #

def test_simulate_price_drop_increases_volume_when_elastic():
    row = {"product": "P1", "mean_price": 10.0, "mean_qty": 100.0,
           "elasticity": -1.5, "r2": 0.8}
    out = whatif.simulate_price_change(row, pct_change=-10)
    # 10% price cut at e=-1.5 → +15% qty
    assert out["pct_qty_change"] == pytest.approx(15.0, rel=0.01)
    assert out["new_qty"] > row["mean_qty"]
    assert out["delta_revenue"] != 0
    assert "delta_low" in out and "delta_high" in out


def test_simulate_price_raise_inelastic():
    row = {"product": "P2", "mean_price": 10.0, "mean_qty": 100.0,
           "elasticity": -0.1, "r2": 0.7}
    out = whatif.simulate_price_change(row, pct_change=10)
    # Price up 10%, qty down only 1% → revenue should rise
    assert out["new_revenue"] > out["baseline_revenue"]


def test_simulate_promo_projects_incremental():
    trend = pd.DataFrame({"period": pd.date_range("2025-01-01", periods=8, freq="W"),
                          "revenue": [1000.0] * 8})
    out = whatif.simulate_promo(trend, lift_pct=10, weeks=4)
    assert out["incremental_per_period"] == pytest.approx(100.0)
    assert out["total_incremental"] == pytest.approx(400.0)


def test_simulate_inventory_reorder():
    stock = pd.DataFrame({
        "product": ["P1"], "on_hand": [50.0], "daily_demand": [10.0],
        "days_to_stockout": [5.0], "risk": ["Critical"],
    })
    out = whatif.simulate_inventory_reorder(stock, "P1", units=100)
    assert out["new_on_hand"] == 150
    assert out["new_days_to_stockout"] == 15.0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not R.HAS_DOCX, reason="python-docx not installed")
def test_build_report_creates_docx():
    from analyst import ingest as I, eda as E, recommend as RC

    csv = "order_date,product_id,customer_id,revenue\n"
    for i in range(20):
        csv += f"2025-01-{(i % 28) + 1:02d},P{(i % 2) + 1},C{i},{100 + i}\n"
    res = I.ingest(csv.encode("utf-8"))
    rep = E.profile(res.df, res.schema, kind=res.kind.kind, role_map=res.kind.role_map)

    rec = RC.Recommendation(
        action="Test action", evidence="Because reasons.",
        confidence="High", impact_estimate="$100",
        audience="all", category="marketing", bandit_arm="test_arm",
    )

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "report.docx"
        path = R.build_report(out_path, ingest_result=res, eda_report=rep,
                              narrative="Synthetic test report.",
                              recommendations=[rec], title="Test Report")
        assert path.exists()
        assert path.stat().st_size > 1000  # non-empty docx
