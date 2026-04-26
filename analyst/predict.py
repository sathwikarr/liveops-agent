"""Prediction engine — churn, stockout horizon, demand forecast.

Reuses agent/forecast.py for time-series so we don't duplicate Prophet plumbing.

Public API:
- `churn_scores(df, role_map, churn_window_days=60)` -> DataFrame[customer, last_seen, days_since, churn_prob, risk]
- `stockout_horizon(df, role_map, days=30)` -> DataFrame[product, on_hand, daily_demand, days_to_stockout, risk]
- `demand_per_segment(df, role_map, periods=14)` -> dict[(region, product) -> forecast DataFrame]

Each function degrades gracefully when prerequisite columns are missing —
returns an empty DataFrame, never raises.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Churn — recency-based logistic-style score
# --------------------------------------------------------------------------- #

def churn_scores(df: pd.DataFrame, role_map: dict,
                 churn_window_days: int = 60) -> pd.DataFrame:
    """Score each customer's churn probability.

    Method: probability rises sigmoidally with `days_since_last_purchase`,
    centered on `churn_window_days`. Customers above 0.5 are at-risk; above
    0.8 considered churned.
    """
    needed = {"customer", "date"}
    if not needed.issubset(role_map):
        return pd.DataFrame()
    cust = role_map["customer"]
    date = role_map["date"]
    if not {cust, date}.issubset(df.columns):
        return pd.DataFrame()

    sub = df[[cust, date]].dropna().copy()
    sub[date] = pd.to_datetime(sub[date], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame()

    snapshot = sub[date].max()
    last = sub.groupby(cust)[date].max().rename("last_seen").reset_index()
    last["days_since"] = (snapshot - last["last_seen"]).dt.days

    # Sigmoid centered on the churn window — gentle slope so the curve is
    # actually informative across nearby days rather than a hard step.
    k = 6.0 / max(churn_window_days, 1)
    last["churn_prob"] = 1 / (1 + np.exp(-k * (last["days_since"] - churn_window_days)))
    last["churn_prob"] = last["churn_prob"].round(3)

    def _risk(p):
        if p >= 0.8: return "Churned"
        if p >= 0.5: return "At-Risk"
        if p >= 0.2: return "Cooling"
        return "Active"

    last["risk"] = last["churn_prob"].apply(_risk)
    last = last.rename(columns={cust: "customer"})
    return last.sort_values("churn_prob", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stockout horizon — current on-hand vs recent velocity
# --------------------------------------------------------------------------- #

def stockout_horizon(df: pd.DataFrame, role_map: dict,
                     lookback_days: int = 14) -> pd.DataFrame:
    """For each product: latest inventory + recent daily demand → days until empty.

    Requires a product role and EITHER an explicit `inventory` column OR a
    `quantity` column (treated as units sold per row).
    """
    if "product" not in role_map:
        return pd.DataFrame()
    product = role_map["product"]
    if product not in df.columns:
        return pd.DataFrame()

    inv_col = role_map.get("inventory")
    qty_col = role_map.get("quantity")
    date_col = role_map.get("date")
    if not (qty_col and date_col):
        return pd.DataFrame()

    sub = df[[product, date_col, qty_col] + ([inv_col] if inv_col and inv_col in df.columns else [])].copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub[qty_col] = pd.to_numeric(sub[qty_col], errors="coerce")
    if inv_col and inv_col in sub.columns:
        sub[inv_col] = pd.to_numeric(sub[inv_col], errors="coerce")
    sub = sub.dropna(subset=[product, date_col, qty_col])
    if sub.empty:
        return pd.DataFrame()

    latest = sub[date_col].max()
    recent_window = sub[sub[date_col] >= latest - pd.Timedelta(days=lookback_days)]

    # Demand per day per product, averaged over the lookback window
    daily = (recent_window.groupby([product, recent_window[date_col].dt.date])[qty_col]
                          .sum().reset_index())
    daily.columns = [product, "_d", "_q"]
    velocity = daily.groupby(product)["_q"].mean().rename("daily_demand")

    if inv_col and inv_col in sub.columns:
        on_hand = sub.sort_values(date_col).groupby(product)[inv_col].last().rename("on_hand")
    else:
        # Without an inventory column we report demand only — not a stockout horizon.
        on_hand = pd.Series(dtype=float, name="on_hand")

    out = pd.concat([on_hand, velocity], axis=1).reset_index().rename(columns={product: "product"})
    if "on_hand" not in out.columns:
        out["on_hand"] = np.nan
    out["days_to_stockout"] = (out["on_hand"] / out["daily_demand"]).round(1)

    def _risk(d):
        if pd.isna(d): return "Unknown"
        if d <= 7: return "Critical"
        if d <= 21: return "Warning"
        return "OK"

    out["risk"] = out["days_to_stockout"].apply(_risk)
    return out.sort_values("days_to_stockout").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Demand per (region, product) — wraps agent/forecast.py
# --------------------------------------------------------------------------- #

def demand_per_segment(df: pd.DataFrame, role_map: dict, periods: int = 14,
                       top_n: int = 6) -> dict:
    """Per-(region, product) demand forecast. Returns {(region, product): DataFrame}.

    Reuses agent.forecast.forecast_per_segment when the schema looks like
    liveops-agent's ops dataset; otherwise builds a compatible projection
    from the role_map.
    """
    try:
        from agent.forecast import forecast_per_segment, recent_top_segments
    except Exception:
        return {}

    needed = {"date", "amount"}
    if not needed.issubset(role_map):
        return {}

    work = df.copy()
    rename = {role_map["date"]: "timestamp", role_map["amount"]: "revenue"}
    if "product" in role_map:
        rename[role_map["product"]] = "product_id"
    if "region" in role_map:
        rename[role_map["region"]] = "region"
    if "quantity" in role_map:
        rename[role_map["quantity"]] = "orders"
    work = work.rename(columns=rename)

    # forecast_per_segment requires region + product_id
    for col in ("region", "product_id"):
        if col not in work.columns:
            work[col] = "all"
    if "orders" not in work.columns:
        work["orders"] = 1
    if "inventory" not in work.columns:
        work["inventory"] = 0

    return forecast_per_segment(work, periods=periods, top_n=top_n)


__all__ = ["churn_scores", "stockout_horizon", "demand_per_segment"]
