"""Recommendation engine — turn analysis outputs into business actions.

Each generator function takes one analysis output (RFM table, elasticity table,
churn scores, …) and returns a list of `Recommendation`s. The page joins them
all into a single feed sorted by confidence × impact.

A `Recommendation` carries:
  - `action`   — what to do, in plain English
  - `evidence` — the numeric facts behind it
  - `confidence` — High / Medium / Low
  - `impact_estimate` — rough $ or % swing if the action lands
  - `audience` — segment / product / region the action targets
  - `category` — pricing / inventory / marketing / retention / merchandising
  - `bandit_arm` — short tag the agent.bandit can use as the arm key

The bandit_arm field threads recs into agent/bandit.py so the existing
"did this work" feedback loop applies — once a user marks a rec done, the
outcome can be logged and future recs of the same type are weighted by
historical success.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Recommendation datatype
# --------------------------------------------------------------------------- #

@dataclass
class Recommendation:
    action: str
    evidence: str
    confidence: str  # "High" | "Medium" | "Low"
    impact_estimate: str
    audience: str
    category: str
    bandit_arm: str
    score: float = 0.0  # internal sort key; higher = better

    def to_dict(self) -> dict:
        return asdict(self)


def _conf(score: float) -> str:
    if score >= 0.7: return "High"
    if score >= 0.4: return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# Generators — one per analysis primitive
# --------------------------------------------------------------------------- #

def from_rfm(rfm_df: pd.DataFrame) -> list[Recommendation]:
    if rfm_df is None or rfm_df.empty:
        return []
    out: list[Recommendation] = []

    at_risk = rfm_df[rfm_df["segment"] == "At-Risk"]
    if len(at_risk) >= 5:
        avg_m = float(at_risk["monetary"].mean())
        score = min(0.95, 0.4 + len(at_risk) / 200)
        out.append(Recommendation(
            action=f"Send a win-back offer to {len(at_risk)} At-Risk customers (10–15% off).",
            evidence=f"They averaged ${avg_m:,.0f} lifetime spend and haven't ordered recently.",
            confidence=_conf(score),
            impact_estimate=f"~${len(at_risk) * avg_m * 0.05:,.0f} recoverable revenue if 5% reactivate.",
            audience=f"{len(at_risk)} customers tagged At-Risk",
            category="retention",
            bandit_arm="winback_email_atrisk",
            score=score,
        ))

    champs = rfm_df[rfm_df["segment"] == "Champions"]
    if len(champs) >= 3:
        avg_m = float(champs["monetary"].mean())
        score = 0.75
        out.append(Recommendation(
            action=f"Invite {len(champs)} Champions to a loyalty / referral program.",
            evidence=f"They average ${avg_m:,.0f} spend and order most frequently.",
            confidence=_conf(score),
            impact_estimate="Referral programs typically lift Champions' LTV by 15–25%.",
            audience=f"{len(champs)} Champion customers",
            category="retention",
            bandit_arm="loyalty_invite_champions",
            score=score,
        ))

    big_lost = rfm_df[rfm_df["segment"] == "Big Spenders Lost"]
    if len(big_lost) >= 2:
        score = 0.65
        out.append(Recommendation(
            action=f"Personal outreach to {len(big_lost)} ex-big-spenders (manual call/email).",
            evidence="Used to spend a lot, now silent — high-value tail-risk worth a real touch.",
            confidence=_conf(score),
            impact_estimate=f"~${big_lost['monetary'].sum() * 0.10:,.0f} possible recovery at 10% reactivation.",
            audience=f"{len(big_lost)} lapsed VIPs",
            category="retention",
            bandit_arm="personal_outreach_lost_vips",
            score=score,
        ))
    return out


def from_elasticity(elast_df: pd.DataFrame) -> list[Recommendation]:
    if elast_df is None or elast_df.empty:
        return []
    out: list[Recommendation] = []
    for _, row in elast_df.iterrows():
        if row["r2"] < 0.3:  # noisy fit — skip
            continue
        e = row["elasticity"]
        # Only act on clearly elastic or clearly inelastic products.
        if e < -1.2:  # very price-sensitive
            score = min(0.9, 0.45 + abs(e) / 5 + row["r2"] * 0.3)
            out.append(Recommendation(
                action=f"Drop {row['product']} price ~10% — price-sensitive (elasticity {e:.2f}).",
                evidence=f"R²={row['r2']:.2f} on {int(row['n'])} observations. "
                         f"Mean price ${row['mean_price']:,.2f}, mean qty {row['mean_qty']:.1f}.",
                confidence=_conf(score),
                impact_estimate=f"Expected volume lift ≈ {abs(e) * 10:.0f}% from a 10% price cut.",
                audience=row["product"],
                category="pricing",
                bandit_arm=f"price_cut_{row['product']}",
                score=score,
            ))
        elif -0.3 < e < 0.0:  # inelastic
            score = min(0.85, 0.4 + row["r2"] * 0.4)
            out.append(Recommendation(
                action=f"Test a 5–10% price increase on {row['product']} — demand barely moves.",
                evidence=f"Elasticity {e:.2f} on {int(row['n'])} observations (R²={row['r2']:.2f}).",
                confidence=_conf(score),
                impact_estimate="Margin gain with minimal volume loss expected.",
                audience=row["product"],
                category="pricing",
                bandit_arm=f"price_raise_{row['product']}",
                score=score,
            ))
    return out


def from_product_matrix(matrix_df: pd.DataFrame) -> list[Recommendation]:
    if matrix_df is None or matrix_df.empty:
        return []
    out: list[Recommendation] = []
    stars = matrix_df[matrix_df["quadrant"] == "Star"].head(3)
    for _, row in stars.iterrows():
        score = 0.7
        out.append(Recommendation(
            action=f"Increase ad spend on {row['product']} — star quadrant.",
            evidence=f"${row['revenue']:,.0f} total revenue ({row['share']:.1%} share), "
                     f"recent growth {row['growth']:+.0%}.",
            confidence=_conf(score),
            impact_estimate="Doubling-down on growing winners typically returns 1.5–3× ad spend.",
            audience=row["product"],
            category="marketing",
            bandit_arm=f"boost_ads_{row['product']}",
            score=score,
        ))

    dogs = matrix_df[matrix_df["quadrant"] == "Dog"]
    if len(dogs) >= 3:
        rev_share = float(dogs["share"].sum())
        score = 0.6
        out.append(Recommendation(
            action=f"Discontinue or clearance {len(dogs)} Dog products.",
            evidence=f"They contribute only {rev_share:.1%} of total revenue and aren't growing.",
            confidence=_conf(score),
            impact_estimate="Frees inventory + ad spend for Stars and Question Marks.",
            audience=", ".join(dogs.head(5)["product"].astype(str)),
            category="merchandising",
            bandit_arm="clearance_dogs",
            score=score,
        ))

    qmarks = matrix_df[matrix_df["quadrant"] == "Question Mark"].head(3)
    for _, row in qmarks.iterrows():
        score = 0.55
        out.append(Recommendation(
            action=f"Run a controlled test promo on {row['product']} — high growth, low share.",
            evidence=f"Growth {row['growth']:+.0%} but only {row['share']:.1%} share — could become a Star.",
            confidence=_conf(score),
            impact_estimate="Limited downside; success here moves it into Star quadrant.",
            audience=row["product"],
            category="marketing",
            bandit_arm=f"test_promo_{row['product']}",
            score=score,
        ))
    return out


def from_basket(basket_df: pd.DataFrame, top_k: int = 3) -> list[Recommendation]:
    if basket_df is None or basket_df.empty:
        return []
    out: list[Recommendation] = []
    for _, row in basket_df.head(top_k).iterrows():
        if row["lift"] < 1.5:
            continue
        score = min(0.85, 0.35 + min(row["lift"] / 5, 0.5))
        out.append(Recommendation(
            action=f"Bundle {row['item_a']} + {row['item_b']} or cross-sell at checkout.",
            evidence=f"Lift {row['lift']:.2f}× expected, {int(row['n_baskets'])} co-purchase baskets, "
                     f"confidence {row['confidence']:.0%}.",
            confidence=_conf(score),
            impact_estimate="Cross-sell bundles typically lift attach rate 8–15%.",
            audience=f"buyers of {row['item_a']}",
            category="merchandising",
            bandit_arm=f"bundle_{row['item_a']}_{row['item_b']}",
            score=score,
        ))
    return out


def from_churn(churn_df: pd.DataFrame) -> list[Recommendation]:
    if churn_df is None or churn_df.empty:
        return []
    out: list[Recommendation] = []
    cooling = churn_df[churn_df["risk"] == "Cooling"]
    at_risk = churn_df[churn_df["risk"] == "At-Risk"]
    if len(cooling) >= 5:
        score = 0.55
        out.append(Recommendation(
            action=f"Trigger a 'we miss you' email to {len(cooling)} cooling customers.",
            evidence="Customers in the 0.2–0.5 churn-prob band — early signal, cheap touch.",
            confidence=_conf(score),
            impact_estimate="Light touch; expected to keep ~20% from sliding into At-Risk.",
            audience=f"{len(cooling)} cooling customers",
            category="retention",
            bandit_arm="email_cooling",
            score=score,
        ))
    if len(at_risk) >= 5:
        score = 0.7
        out.append(Recommendation(
            action=f"Send a 15% offer + free-ship to {len(at_risk)} At-Risk customers.",
            evidence=f"Avg days since last purchase: {at_risk['days_since'].mean():.0f}.",
            confidence=_conf(score),
            impact_estimate="Discount-led win-backs typically reactivate 10–20% within 30 days.",
            audience=f"{len(at_risk)} At-Risk customers",
            category="retention",
            bandit_arm="discount_winback_atrisk",
            score=score,
        ))
    return out


def from_stockout(stock_df: pd.DataFrame) -> list[Recommendation]:
    if stock_df is None or stock_df.empty:
        return []
    out: list[Recommendation] = []
    crit = stock_df[stock_df["risk"] == "Critical"]
    warn = stock_df[stock_df["risk"] == "Warning"]
    for _, row in crit.iterrows():
        score = 0.9
        out.append(Recommendation(
            action=f"Reorder {row['product']} now — stockout in {row['days_to_stockout']:.0f} days.",
            evidence=f"On hand {row['on_hand']:.0f}, daily demand {row['daily_demand']:.1f}.",
            confidence=_conf(score),
            impact_estimate=f"Lost-sales risk ≈ {row['daily_demand'] * 7:.0f} units/wk after stockout.",
            audience=row["product"],
            category="inventory",
            bandit_arm=f"reorder_{row['product']}",
            score=score,
        ))
    for _, row in warn.iterrows():
        score = 0.65
        out.append(Recommendation(
            action=f"Plan reorder for {row['product']} (~{row['days_to_stockout']:.0f}d cover).",
            evidence=f"On hand {row['on_hand']:.0f}, daily demand {row['daily_demand']:.1f}.",
            confidence=_conf(score),
            impact_estimate="Avoids the scramble window if demand ticks up.",
            audience=row["product"],
            category="inventory",
            bandit_arm=f"plan_reorder_{row['product']}",
            score=score,
        ))
    return out


def from_seasonality(seasonality, role_map: dict) -> list[Recommendation]:
    """Promo-windowing based on weekly/monthly peaks from analyst.eda."""
    if seasonality is None:
        return []
    out: list[Recommendation] = []
    if getattr(seasonality, "has_weekly", False) and seasonality.peak_weekday:
        score = 0.6
        out.append(Recommendation(
            action=f"Schedule promotions to land on {seasonality.peak_weekday}.",
            evidence=f"Weekly peak ratio {seasonality.weekly_strength:.2f}× — "
                     f"{seasonality.peak_weekday} is your strongest day.",
            confidence=_conf(score),
            impact_estimate=f"Promo attach on peak days typically lifts daily revenue ~{(seasonality.weekly_strength - 1) * 50:.0f}%.",
            audience="all customers",
            category="marketing",
            bandit_arm="promo_weekly_peak",
            score=score,
        ))
    if getattr(seasonality, "has_monthly", False) and seasonality.peak_month:
        score = 0.55
        out.append(Recommendation(
            action=f"Plan major campaigns around {seasonality.peak_month}.",
            evidence=f"Monthly peak ratio {seasonality.monthly_strength:.2f}× — "
                     f"{seasonality.peak_month} is the strongest month.",
            confidence=_conf(score),
            impact_estimate="Aligning ad budget with seasonal peak months is the single highest-ROI calendar shift.",
            audience="all customers",
            category="marketing",
            bandit_arm="campaign_monthly_peak",
            score=score,
        ))
    return out


# --------------------------------------------------------------------------- #
# Top-level: combine + rank
# --------------------------------------------------------------------------- #

def generate(*,
             rfm_df: Optional[pd.DataFrame] = None,
             elasticity_df: Optional[pd.DataFrame] = None,
             matrix_df: Optional[pd.DataFrame] = None,
             basket_df: Optional[pd.DataFrame] = None,
             churn_df: Optional[pd.DataFrame] = None,
             stockout_df: Optional[pd.DataFrame] = None,
             seasonality=None,
             role_map: Optional[dict] = None,
             bandit_username: Optional[str] = None) -> list[Recommendation]:
    """Run every generator on whatever inputs are provided, sort by score.

    If `bandit_username` is given, weight each rec's score by the bandit's
    historical success on that arm so past wins bubble up.
    """
    role_map = role_map or {}
    recs: list[Recommendation] = []
    recs += from_rfm(rfm_df) if rfm_df is not None else []
    recs += from_elasticity(elasticity_df) if elasticity_df is not None else []
    recs += from_product_matrix(matrix_df) if matrix_df is not None else []
    recs += from_basket(basket_df) if basket_df is not None else []
    recs += from_churn(churn_df) if churn_df is not None else []
    recs += from_stockout(stockout_df) if stockout_df is not None else []
    recs += from_seasonality(seasonality, role_map)

    if bandit_username:
        try:
            from agent import bandit
            for r in recs:
                s, f = bandit._arm_counts(r.bandit_arm, bandit_username)
                if s + f >= 3:
                    # Beta-Bernoulli posterior mean as a soft prior on score
                    posterior = (s + 1) / (s + f + 2)
                    r.score = 0.7 * r.score + 0.3 * posterior
                    r.confidence = _conf(r.score)
        except Exception:
            pass

    recs.sort(key=lambda r: r.score, reverse=True)
    return recs


__all__ = ["Recommendation", "generate",
           "from_rfm", "from_elasticity", "from_product_matrix",
           "from_basket", "from_churn", "from_stockout", "from_seasonality"]
