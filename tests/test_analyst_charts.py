"""Tests for analyst.charts — every builder must return a Plotly Figure
without exceptions, even on degenerate input. We don't snapshot pixel
output; we assert shape (trace counts, axis titles, layout title)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from analyst import charts


# --------------------------------------------------------------------------- #
# Revenue trend
# --------------------------------------------------------------------------- #

def test_revenue_trend_returns_figure_with_two_traces():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=12, freq="W"),
        "rev": np.arange(12) * 100.0,
    })
    fig = charts.revenue_trend(df, "date", "rev")
    assert isinstance(fig, go.Figure)
    # bar + moving average line
    assert len(fig.data) == 2
    assert "Revenue trend" in fig.layout.title.text


def test_revenue_trend_handles_empty():
    fig = charts.revenue_trend(pd.DataFrame(), "date", "rev")
    assert isinstance(fig, go.Figure)
    assert "no data" in fig.layout.title.text


def test_revenue_trend_handles_missing_columns():
    df = pd.DataFrame({"x": [1, 2]})
    fig = charts.revenue_trend(df, "date", "rev")
    assert "no data" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# RFM
# --------------------------------------------------------------------------- #

def test_rfm_scatter_basic():
    rfm = pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "recency": [10, 200, 5, 90],
        "frequency": [4, 1, 8, 2],
        "monetary": [400, 50, 1200, 200],
        "segment": ["Loyal", "Lost", "Champions", "Average"],
    })
    fig = charts.rfm_scatter(rfm)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_rfm_scatter_handles_empty():
    fig = charts.rfm_scatter(pd.DataFrame())
    assert "no data" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# Product matrix
# --------------------------------------------------------------------------- #

def test_product_matrix_basic():
    pm = pd.DataFrame({
        "product": ["A", "B", "C", "D"],
        "revenue": [1000, 500, 200, 100],
        "growth": [0.4, -0.1, 0.3, -0.2],
        "units": [50, 30, 20, 10],
        "quadrant": ["Star", "Cash Cow", "Question Mark", "Dog"],
    })
    fig = charts.product_matrix(pm)
    assert isinstance(fig, go.Figure)
    # plotly express scatter colored by quadrant produces 1 trace per category
    assert len(fig.data) >= 1


def test_product_matrix_handles_empty():
    fig = charts.product_matrix(pd.DataFrame())
    assert "no data" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# Elasticity
# --------------------------------------------------------------------------- #

def test_elasticity_scatter_renders():
    p = pd.Series([10, 20, 30, 40, 50])
    q = pd.Series([100, 50, 33, 25, 20])
    fig = charts.elasticity_scatter(p, q, slope=-1.0, intercept=4.6, r2=0.95)
    assert len(fig.data) == 2  # observed + fit


def test_elasticity_scatter_handles_few_points():
    p = pd.Series([10])
    q = pd.Series([100])
    fig = charts.elasticity_scatter(p, q, slope=0, intercept=0, r2=0)
    assert "not enough" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# Churn
# --------------------------------------------------------------------------- #

def test_churn_distribution_renders():
    df = pd.DataFrame({"churn_prob": np.linspace(0, 1, 100)})
    fig = charts.churn_distribution(df)
    assert len(fig.data) == 1
    # Range pinned to [0, 1]
    assert tuple(fig.layout.xaxis.range) == (0, 1)


def test_churn_distribution_handles_empty():
    fig = charts.churn_distribution(pd.DataFrame())
    assert "no data" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# Calendar gantt
# --------------------------------------------------------------------------- #

def test_calendar_gantt_renders():
    df = pd.DataFrame({
        "week_start": [date(2026, 5, 4), date(2026, 5, 11)],
        "action": ["Reorder X", "Promote Y"],
        "category": ["inventory", "marketing"],
        "audience": ["all", "all"],
        "confidence": ["High", "High"],
        "score": [0.9, 0.8],
    })
    fig = charts.calendar_gantt(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_calendar_gantt_handles_empty():
    fig = charts.calendar_gantt(pd.DataFrame())
    assert "no data" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# Heatmaps
# --------------------------------------------------------------------------- #

def test_correlation_heatmap_renders():
    corr = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["a", "b"], columns=["a", "b"],
    )
    fig = charts.correlation_heatmap(corr)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_cohort_heatmap_renders():
    cohort = pd.DataFrame(
        [[1.0, 0.5, 0.3], [1.0, 0.6, np.nan], [1.0, np.nan, np.nan]],
        index=["2026-01", "2026-02", "2026-03"], columns=[0, 1, 2],
    )
    fig = charts.cohort_heatmap(cohort)
    assert len(fig.data) == 1


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_registry_contains_all_builders():
    expected = {"revenue_trend", "rfm_scatter", "product_matrix",
                "elasticity_scatter", "churn_distribution", "calendar_gantt",
                "correlation_heatmap", "cohort_heatmap"}
    assert expected.issubset(charts.REGISTRY.keys())


def test_palette_has_known_segments():
    for k in ["Star", "Cash Cow", "Champions", "Lost"]:
        assert k in charts.PALETTE
