"""Analysis library — RFM, cohorts, basket, elasticity, product matrix.

Each function takes a clean DataFrame + a `role_map` (from `analyst.ingest`)
so it can find the right columns regardless of the user's naming. Every
function returns a `pandas` DataFrame; the Streamlit page picks the right
chart type per result.

Deliberately small + composable — no model objects, no internal state. The
`recommend` layer (Stage 6) consumes these outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _resolve(role_map: dict, role: str, fallback_cols: list[str]) -> Optional[str]:
    if role in role_map:
        return role_map[role]
    for c in fallback_cols:
        if c in role_map.values():
            continue
        return c
    return None


def _need(role_map: dict, *roles: str) -> Optional[str]:
    for r in roles:
        if r not in role_map:
            return r
    return None


# --------------------------------------------------------------------------- #
# RFM (Recency / Frequency / Monetary)
# --------------------------------------------------------------------------- #

def rfm(df: pd.DataFrame, role_map: dict, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Compute RFM scores per customer + assign segment labels.

    Returns columns: customer, recency_days, frequency, monetary,
    R, F, M, rfm_score, segment.
    """
    missing = _need(role_map, "customer", "date", "amount")
    if missing:
        return pd.DataFrame()

    cust = role_map["customer"]
    date = role_map["date"]
    amt = role_map["amount"]
    if not {cust, date, amt}.issubset(df.columns):
        return pd.DataFrame()

    sub = df[[cust, date, amt]].dropna()
    sub[date] = pd.to_datetime(sub[date], errors="coerce")
    sub[amt] = pd.to_numeric(sub[amt], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame()

    snapshot = pd.Timestamp(as_of) if as_of is not None else sub[date].max() + pd.Timedelta(days=1)
    grouped = sub.groupby(cust).agg(
        recency_days=(date, lambda s: (snapshot - s.max()).days),
        frequency=(date, "count"),
        monetary=(amt, "sum"),
    ).reset_index().rename(columns={cust: "customer"})

    def _score(s: pd.Series, ascending: bool) -> pd.Series:
        # Quartile bins, 1..4. Higher score = better.
        try:
            q = pd.qcut(s.rank(method="first"), q=4, labels=[1, 2, 3, 4])
            return q.astype(int) if ascending else (5 - q.astype(int))
        except ValueError:
            return pd.Series([2] * len(s), index=s.index)

    grouped["R"] = _score(grouped["recency_days"], ascending=False)  # lower recency = better
    grouped["F"] = _score(grouped["frequency"], ascending=True)
    grouped["M"] = _score(grouped["monetary"], ascending=True)
    grouped["rfm_score"] = grouped["R"] * 100 + grouped["F"] * 10 + grouped["M"]

    def _label(row) -> str:
        r, f, m = row["R"], row["F"], row["M"]
        if r >= 4 and f >= 4 and m >= 4: return "Champions"
        if r >= 3 and f >= 3 and m >= 3: return "Loyal"
        if r >= 4 and f <= 2:            return "New"
        if r <= 2 and f >= 3:            return "At-Risk"
        if r <= 2 and f <= 2 and m >= 3: return "Big Spenders Lost"
        if r <= 1:                       return "Lost"
        return "Average"

    grouped["segment"] = grouped.apply(_label, axis=1)
    return grouped.sort_values("rfm_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cohort retention (acquisition month → activity per month-since)
# --------------------------------------------------------------------------- #

def cohort_retention(df: pd.DataFrame, role_map: dict) -> pd.DataFrame:
    """Return a cohort matrix: rows = acquisition month, cols = months since."""
    missing = _need(role_map, "customer", "date")
    if missing:
        return pd.DataFrame()

    cust, date = role_map["customer"], role_map["date"]
    if not {cust, date}.issubset(df.columns):
        return pd.DataFrame()
    sub = df[[cust, date]].dropna().copy()
    sub[date] = pd.to_datetime(sub[date], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame()

    sub["_month"] = sub[date].dt.to_period("M")
    first = sub.groupby(cust)["_month"].min().rename("_cohort")
    sub = sub.join(first, on=cust)
    sub["_offset"] = (sub["_month"] - sub["_cohort"]).apply(
        lambda x: x.n if hasattr(x, "n") else 0
    )

    pivot = (sub.groupby(["_cohort", "_offset"])[cust].nunique()
                .unstack(fill_value=0).sort_index())
    if pivot.empty:
        return pivot

    base = pivot.iloc[:, 0].replace(0, np.nan)
    retention = pivot.div(base, axis=0).round(3)
    retention.index = retention.index.astype(str)
    return retention


# --------------------------------------------------------------------------- #
# Market basket — simple co-purchase counts + lift
# --------------------------------------------------------------------------- #

def market_basket(df: pd.DataFrame, role_map: dict, top_n: int = 20,
                  basket_key: Optional[str] = None) -> pd.DataFrame:
    """Co-purchase pairs sorted by lift.

    A "basket" is rows sharing the same `basket_key` (or, if missing,
    same customer+date). Returns: item_a, item_b, support, confidence, lift.
    """
    missing = _need(role_map, "product")
    if missing:
        return pd.DataFrame()
    product = role_map["product"]

    if basket_key and basket_key in df.columns:
        bk = basket_key
    elif "customer" in role_map and "date" in role_map:
        bk = "_basket"
        df = df.copy()
        df[bk] = df[role_map["customer"]].astype(str) + "|" + df[role_map["date"]].astype(str)
    else:
        return pd.DataFrame()

    sub = df[[bk, product]].dropna().drop_duplicates()
    baskets = sub.groupby(bk)[product].apply(set)
    n_baskets = len(baskets)
    if n_baskets < 5:
        return pd.DataFrame()

    item_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for items in baskets:
        items = sorted(map(str, items))
        for it in items:
            item_counts[it] = item_counts.get(it, 0) + 1
        for a, b in combinations(items, 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    rows = []
    for (a, b), c in pair_counts.items():
        sup = c / n_baskets
        conf = c / max(item_counts[a], 1)
        # lift = P(B|A) / P(B)
        pb = item_counts[b] / n_baskets
        lift = conf / pb if pb else 0.0
        rows.append({"item_a": a, "item_b": b, "support": round(sup, 4),
                     "confidence": round(conf, 4), "lift": round(lift, 3),
                     "n_baskets": c})
    out = pd.DataFrame(rows).sort_values(["lift", "support"], ascending=False).head(top_n)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Price elasticity — log-log regression slope per product
# --------------------------------------------------------------------------- #

def elasticity(df: pd.DataFrame, role_map: dict,
               price_col: Optional[str] = None) -> pd.DataFrame:
    """Estimate price elasticity per product.

    elasticity = d(ln Q) / d(ln P). Negative + significant ⇒ price-sensitive.
    Returns: product, n, mean_price, mean_qty, elasticity, r2.
    """
    missing = _need(role_map, "product", "quantity")
    if missing:
        return pd.DataFrame()
    product = role_map["product"]
    qty = role_map["quantity"]
    price = price_col or _resolve(role_map, "price", [])
    if not price:
        # Try to derive from amount/quantity
        if "amount" in role_map and qty in df.columns:
            df = df.copy()
            df["_unit_price"] = pd.to_numeric(df[role_map["amount"]], errors="coerce") / \
                                pd.to_numeric(df[qty], errors="coerce").replace(0, np.nan)
            price = "_unit_price"
        else:
            return pd.DataFrame()
    if price not in df.columns:
        return pd.DataFrame()

    rows = []
    for p, sub in df.groupby(product):
        x = pd.to_numeric(sub[price], errors="coerce")
        y = pd.to_numeric(sub[qty], errors="coerce")
        m = x.notna() & y.notna() & (x > 0) & (y > 0)
        if m.sum() < 10 or x[m].nunique() < 3:
            continue
        lx = np.log(x[m].to_numpy())
        ly = np.log(y[m].to_numpy())
        slope, intercept = np.polyfit(lx, ly, 1)
        ss_res = ((ly - (slope * lx + intercept)) ** 2).sum()
        ss_tot = ((ly - ly.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        rows.append({
            "product": str(p), "n": int(m.sum()),
            "mean_price": round(float(x[m].mean()), 3),
            "mean_qty": round(float(y[m].mean()), 3),
            "elasticity": round(float(slope), 3),
            "r2": round(float(r2), 3),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("elasticity").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Product performance matrix — revenue × growth quadrant
# --------------------------------------------------------------------------- #

def product_matrix(df: pd.DataFrame, role_map: dict) -> pd.DataFrame:
    """Per product: total revenue, share, recent growth, quadrant.

    Quadrants: Star (high rev + growing), Cash Cow (high rev + flat),
    Question Mark (low rev + growing), Dog (low rev + flat/declining).
    """
    missing = _need(role_map, "product", "amount", "date")
    if missing:
        return pd.DataFrame()
    product = role_map["product"]
    amt = role_map["amount"]
    date = role_map["date"]
    if not {product, amt, date}.issubset(df.columns):
        return pd.DataFrame()

    sub = df[[product, amt, date]].dropna().copy()
    sub[amt] = pd.to_numeric(sub[amt], errors="coerce")
    sub[date] = pd.to_datetime(sub[date], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame()

    cutoff = sub[date].max() - (sub[date].max() - sub[date].min()) / 2
    recent = sub[sub[date] >= cutoff].groupby(product)[amt].sum()
    older = sub[sub[date] < cutoff].groupby(product)[amt].sum()
    total = sub.groupby(product)[amt].sum()

    out = pd.DataFrame({
        "revenue": total,
        "recent_revenue": recent,
        "older_revenue": older,
    }).fillna(0.0)
    out["share"] = (out["revenue"] / out["revenue"].sum()).round(4)
    out["growth"] = ((out["recent_revenue"] - out["older_revenue"])
                     / out["older_revenue"].replace(0, np.nan)).round(3)
    out["growth"] = out["growth"].fillna(0.0)

    rev_med = out["revenue"].median()

    def _quadrant(row) -> str:
        hi_rev = row["revenue"] >= rev_med
        growing = row["growth"] > 0.05
        if hi_rev and growing: return "Star"
        if hi_rev and not growing: return "Cash Cow"
        if not hi_rev and growing: return "Question Mark"
        return "Dog"

    out["quadrant"] = out.apply(_quadrant, axis=1)
    out = out.reset_index().rename(columns={product: "product"})
    return out.sort_values("revenue", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trend — period-over-period revenue summary
# --------------------------------------------------------------------------- #

def revenue_trend(df: pd.DataFrame, role_map: dict, freq: str = "W") -> pd.DataFrame:
    """Aggregate revenue at the requested frequency + roll-up MoM/YoY where possible."""
    missing = _need(role_map, "date", "amount")
    if missing:
        return pd.DataFrame()
    date, amt = role_map["date"], role_map["amount"]
    if not {date, amt}.issubset(df.columns):
        return pd.DataFrame()
    sub = df[[date, amt]].dropna().copy()
    sub[date] = pd.to_datetime(sub[date], errors="coerce")
    sub[amt] = pd.to_numeric(sub[amt], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame()

    s = sub.set_index(date)[amt].resample(freq).sum().rename("revenue").to_frame()
    s["pct_change"] = s["revenue"].pct_change().round(3)
    s["rolling_4"] = s["revenue"].rolling(4, min_periods=1).mean().round(2)
    s.index.name = "period"
    return s.reset_index()


__all__ = ["rfm", "cohort_retention", "market_basket", "elasticity",
           "product_matrix", "revenue_trend"]
