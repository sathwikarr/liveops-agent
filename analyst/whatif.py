"""What-if simulator — apply scenarios on top of analysis outputs.

Public API:
- `simulate_price_change(elast_row, pct_change)` -> dict
    Predict revenue/volume swing using the elasticity table from
    analyst.analysis.elasticity. Reports point estimate + confidence band.

- `simulate_promo(rev_trend_df, lift_pct, weeks)` -> dict
    Apply a flat lift % to the trailing weeks of a revenue trend, project
    the delta vs no-promo baseline.

- `simulate_inventory_reorder(stock_df, product, units)` -> dict
    Recompute days_to_stockout if `units` are added to current on-hand.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def simulate_price_change(elast_row: dict, pct_change: float) -> dict:
    """Predict the volume + revenue impact of a price move.

    elast_row should look like one row from analyst.analysis.elasticity:
        {"product": ..., "mean_price": ..., "mean_qty": ..., "elasticity": ..., "r2": ...}

    pct_change is signed: -10 means a 10% price cut.
    """
    e = float(elast_row["elasticity"])
    p0 = float(elast_row["mean_price"])
    q0 = float(elast_row["mean_qty"])
    r2 = float(elast_row.get("r2", 0.5))

    new_price = p0 * (1 + pct_change / 100.0)
    pct_q_change = e * (pct_change / 100.0)  # elasticity definition
    new_qty = q0 * (1 + pct_q_change)
    new_rev = new_price * new_qty
    base_rev = p0 * q0

    # Confidence band: scale 1σ by (1 - r2) and the magnitude of the move.
    band = abs(new_rev - base_rev) * (1 - r2) * 0.5
    return {
        "product": elast_row.get("product"),
        "pct_price_change": round(pct_change, 2),
        "pct_qty_change": round(pct_q_change * 100, 2),
        "new_price": round(new_price, 2),
        "new_qty": round(new_qty, 2),
        "new_revenue": round(new_rev, 2),
        "baseline_revenue": round(base_rev, 2),
        "delta_revenue": round(new_rev - base_rev, 2),
        "delta_low": round(new_rev - base_rev - band, 2),
        "delta_high": round(new_rev - base_rev + band, 2),
        "confidence_r2": round(r2, 3),
    }


def simulate_promo(trend_df: pd.DataFrame, lift_pct: float, weeks: int = 4) -> dict:
    """Apply a flat % lift to the next `weeks` periods, project incremental revenue."""
    if trend_df is None or trend_df.empty or "revenue" not in trend_df.columns:
        return {"error": "Need a revenue trend with at least one period."}
    avg = float(trend_df["revenue"].tail(weeks * 2).mean())
    incremental_per_period = avg * (lift_pct / 100.0)
    total = incremental_per_period * weeks
    return {
        "weeks": int(weeks),
        "lift_pct": round(lift_pct, 2),
        "baseline_avg_per_period": round(avg, 2),
        "incremental_per_period": round(incremental_per_period, 2),
        "total_incremental": round(total, 2),
    }


def simulate_inventory_reorder(stock_df: pd.DataFrame, product: str, units: int) -> dict:
    """Recompute days_to_stockout for a product after adding `units` on-hand."""
    if stock_df is None or stock_df.empty:
        return {"error": "Need a stockout-horizon table."}
    sub = stock_df[stock_df["product"].astype(str) == str(product)]
    if sub.empty:
        return {"error": f"Product {product!r} not found in stock table."}
    row = sub.iloc[0]
    new_on_hand = float(row["on_hand"]) + float(units)
    daily = float(row["daily_demand"]) if pd.notna(row["daily_demand"]) and row["daily_demand"] > 0 else None
    new_days = round(new_on_hand / daily, 1) if daily else None
    return {
        "product": product,
        "old_on_hand": float(row["on_hand"]),
        "new_on_hand": new_on_hand,
        "daily_demand": daily,
        "old_days_to_stockout": float(row["days_to_stockout"]) if pd.notna(row["days_to_stockout"]) else None,
        "new_days_to_stockout": new_days,
    }


__all__ = ["simulate_price_change", "simulate_promo", "simulate_inventory_reorder"]
