"""Plotly chart factory for the analyst workbench.

Every builder takes a DataFrame (and a small set of params) and returns a
plotly.graph_objects.Figure. The Streamlit page calls these directly via
`st.plotly_chart`. The Pinboard module persists `(builder_name, params)`
tuples and re-renders them later — so every builder must accept JSON-safe
params and be deterministic given the same inputs.

Public API:
- revenue_trend(df, date_col, rev_col, freq="W")
- rfm_scatter(rfm_df) — expects RFM table from analyst.analysis.rfm
- product_matrix(matrix_df) — expects analyst.analysis.product_matrix output
- elasticity_scatter(price, qty, slope, intercept, r2)
- churn_distribution(churn_df, prob_col="churn_prob")
- calendar_gantt(calendar_df) — from analyst.calendar.build_calendar
- correlation_heatmap(corr_df)
- cohort_heatmap(cohort_df) — from analyst.analysis.cohort_retention

Each builder also exposes a small spec helper (`_spec_*`) used by the
Pinboard. We keep specs explicit instead of pickling Figures — that way
session reloads still produce live charts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Consistent color palette — picked for distinguishable hues at small sizes.
PALETTE = {
    "Star": "#22c55e",
    "Cash Cow": "#3b82f6",
    "Question Mark": "#f59e0b",
    "Dog": "#ef4444",
    "Champions": "#16a34a",
    "Loyal": "#22c55e",
    "New": "#06b6d4",
    "At-Risk": "#f97316",
    "Big Spenders Lost": "#ef4444",
    "Lost": "#7f1d1d",
    "Average": "#94a3b8",
    "Critical": "#ef4444",
    "Warning": "#f59e0b",
    "OK": "#22c55e",
}


def _layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15)),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7")
    return fig


# --------------------------------------------------------------------------- #
# Revenue trend
# --------------------------------------------------------------------------- #

def revenue_trend(df: pd.DataFrame, date_col: str, rev_col: str,
                  freq: str = "W") -> go.Figure:
    """Time series of revenue with a 4-period rolling mean overlay."""
    if df is None or df.empty or date_col not in df or rev_col not in df:
        return _layout(go.Figure(), "Revenue trend (no data)")

    s = df[[date_col, rev_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col]).set_index(date_col).sort_index()
    s = s[rev_col].resample(freq).sum()
    rolling = s.rolling(window=4, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.index, y=s.values, name="Revenue",
                         marker_color="#3b82f6", opacity=0.55))
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values,
                             name="4-period MA", mode="lines",
                             line=dict(color="#1e293b", width=2.5)))
    return _layout(fig, f"Revenue trend ({freq})")


# --------------------------------------------------------------------------- #
# RFM scatter
# --------------------------------------------------------------------------- #

def rfm_scatter(rfm_df: pd.DataFrame) -> go.Figure:
    """2D scatter: x=Recency, y=Monetary, size=Frequency, color=Segment.

    Tolerates both the canonical analyst.analysis output column names
    (`recency_days`, `frequency`, `monetary`, `segment`) and shorter
    `recency` aliases used in some test fixtures.
    """
    if rfm_df is None or rfm_df.empty:
        return _layout(go.Figure(), "RFM segments (no data)")

    df = rfm_df.copy()
    rec = "recency_days" if "recency_days" in df.columns else "recency"
    needed = {rec, "frequency", "monetary", "segment"}
    if not needed.issubset(df.columns):
        return _layout(go.Figure(), "RFM segments (missing columns)")

    fig = px.scatter(
        df, x=rec, y="monetary", size="frequency",
        color="segment", color_discrete_map=PALETTE,
        size_max=28, hover_data=[rec, "frequency", "monetary"],
        log_y=True,
    )
    fig.update_xaxes(title="Recency (days since last order)")
    fig.update_yaxes(title="Monetary value (log)")
    return _layout(fig, "RFM segmentation")


# --------------------------------------------------------------------------- #
# Product matrix (BCG quadrants)
# --------------------------------------------------------------------------- #

def product_matrix(matrix_df: pd.DataFrame) -> go.Figure:
    """4-color quadrant scatter: revenue vs growth, sized by units (or revenue
    fallback when units aren't available — analyst.analysis.product_matrix
    doesn't compute units, only revenue/growth/quadrant)."""
    base = {"product", "revenue", "growth", "quadrant"}
    if matrix_df is None or matrix_df.empty or not base.issubset(matrix_df.columns):
        return _layout(go.Figure(), "Product matrix (no data)")

    df = matrix_df.copy()
    size_col = "units" if "units" in df.columns else "revenue"

    fig = px.scatter(
        df, x="revenue", y="growth", size=size_col, color="quadrant",
        color_discrete_map=PALETTE, hover_name="product",
        size_max=40,
    )
    # Quadrant guides: median revenue line, zero growth line.
    rev_med = float(matrix_df["revenue"].median())
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    fig.add_vline(x=rev_med, line_dash="dot", line_color="#94a3b8")
    fig.update_xaxes(title="Revenue", type="log")
    fig.update_yaxes(title="Growth (period-over-period)", tickformat=".0%")
    return _layout(fig, "Product matrix (Star / Cash Cow / Question / Dog)")


# --------------------------------------------------------------------------- #
# Elasticity log-log
# --------------------------------------------------------------------------- #

def elasticity_scatter(price: pd.Series, qty: pd.Series,
                       slope: float, intercept: float,
                       r2: float) -> go.Figure:
    """Log-log price vs quantity scatter with the OLS fit line overlaid."""
    if price is None or qty is None or len(price) == 0:
        return _layout(go.Figure(), "Price elasticity (no data)")

    p = pd.to_numeric(price, errors="coerce")
    q = pd.to_numeric(qty, errors="coerce")
    mask = (p > 0) & (q > 0)
    p, q = p[mask], q[mask]
    if len(p) < 2:
        return _layout(go.Figure(), "Price elasticity (not enough points)")

    lp, lq = np.log(p), np.log(q)
    fit_x = np.linspace(lp.min(), lp.max(), 50)
    fit_y = intercept + slope * fit_x

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.exp(lp), y=np.exp(lq), mode="markers",
                             name="Observed", marker=dict(color="#3b82f6", size=8, opacity=0.7)))
    fig.add_trace(go.Scatter(x=np.exp(fit_x), y=np.exp(fit_y), mode="lines",
                             name=f"Fit (e={slope:.2f}, R²={r2:.2f})",
                             line=dict(color="#ef4444", width=2.5, dash="dash")))
    fig.update_xaxes(title="Price (log)", type="log")
    fig.update_yaxes(title="Quantity (log)", type="log")
    return _layout(fig, "Price elasticity")


# --------------------------------------------------------------------------- #
# Churn distribution
# --------------------------------------------------------------------------- #

def churn_distribution(churn_df: pd.DataFrame,
                       prob_col: str = "churn_prob") -> go.Figure:
    """Histogram of churn probability with risk band shading."""
    if churn_df is None or churn_df.empty or prob_col not in churn_df.columns:
        return _layout(go.Figure(), "Churn distribution (no data)")

    p = pd.to_numeric(churn_df[prob_col], errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=p, nbinsx=20, marker_color="#3b82f6",
                               opacity=0.85, name="Customers"))
    # Risk bands as shaded rectangles.
    fig.add_vrect(x0=0, x1=0.3, fillcolor="#22c55e", opacity=0.08, line_width=0,
                  annotation_text="Healthy", annotation_position="top left")
    fig.add_vrect(x0=0.3, x1=0.7, fillcolor="#f59e0b", opacity=0.08, line_width=0,
                  annotation_text="At-Risk", annotation_position="top left")
    fig.add_vrect(x0=0.7, x1=1.0, fillcolor="#ef4444", opacity=0.10, line_width=0,
                  annotation_text="Likely churn", annotation_position="top left")
    fig.update_xaxes(title="Churn probability", range=[0, 1])
    fig.update_yaxes(title="Customer count")
    return _layout(fig, "Churn risk distribution")


# --------------------------------------------------------------------------- #
# Calendar Gantt
# --------------------------------------------------------------------------- #

def calendar_gantt(calendar_df: pd.DataFrame) -> go.Figure:
    """Gantt-style timeline of weekly actions, color-coded by category."""
    if calendar_df is None or calendar_df.empty or "week_start" not in calendar_df:
        return _layout(go.Figure(), "Action calendar (no data)")

    df = calendar_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["week_end"] = df["week_start"] + pd.Timedelta(days=6)
    fig = px.timeline(df, x_start="week_start", x_end="week_end",
                      y="category", color="category",
                      hover_name="action",
                      hover_data=["audience", "confidence", "score"])
    fig.update_yaxes(autorange="reversed", title="Category")
    fig.update_xaxes(title="Week")
    return _layout(fig, "Action calendar", height=380)


# --------------------------------------------------------------------------- #
# Correlation heatmap
# --------------------------------------------------------------------------- #

def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Pearson correlation heatmap with a diverging colorscale."""
    if corr_df is None or corr_df.empty:
        return _layout(go.Figure(), "Correlations (no data)")
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values, x=list(corr_df.columns), y=list(corr_df.index),
        zmin=-1, zmax=1, colorscale="RdBu",
        colorbar=dict(title="r"),
    ))
    return _layout(fig, "Correlations", height=380)


# --------------------------------------------------------------------------- #
# Cohort retention heatmap
# --------------------------------------------------------------------------- #

def cohort_heatmap(cohort_df: pd.DataFrame) -> go.Figure:
    """Triangle-shaped retention matrix; rows = cohort month, cols = age."""
    if cohort_df is None or cohort_df.empty:
        return _layout(go.Figure(), "Cohort retention (no data)")
    z = cohort_df.values
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[str(c) for c in cohort_df.columns],
        y=[str(i) for i in cohort_df.index],
        zmin=0, zmax=1, colorscale="Blues",
        colorbar=dict(title="Retention"),
    ))
    fig.update_xaxes(title="Months since first order")
    fig.update_yaxes(title="Cohort", autorange="reversed")
    return _layout(fig, "Cohort retention", height=380)


# --------------------------------------------------------------------------- #
# Pinboard-friendly registry
# --------------------------------------------------------------------------- #

REGISTRY = {
    "revenue_trend": revenue_trend,
    "rfm_scatter": rfm_scatter,
    "product_matrix": product_matrix,
    "elasticity_scatter": elasticity_scatter,
    "churn_distribution": churn_distribution,
    "calendar_gantt": calendar_gantt,
    "correlation_heatmap": correlation_heatmap,
    "cohort_heatmap": cohort_heatmap,
}


__all__ = ["revenue_trend", "rfm_scatter", "product_matrix",
           "elasticity_scatter", "churn_distribution", "calendar_gantt",
           "correlation_heatmap", "cohort_heatmap", "REGISTRY", "PALETTE"]
