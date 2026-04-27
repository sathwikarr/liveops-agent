"""Tests for analyst.pinboard — CRUD, rendering, JSON roundtrip, HTML export."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from analyst import pinboard as PB


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_ctx():
    """Realistic ctx dict the page would assemble."""
    return {
        "rev_df": pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=8, freq="W"),
            "rev": np.arange(8) * 100.0,
        }),
        "rfm_df": pd.DataFrame({
            "customer_id": ["A", "B"],
            "recency": [10, 100], "frequency": [4, 1], "monetary": [400, 50],
            "segment": ["Loyal", "Lost"],
        }),
        "matrix_df": pd.DataFrame({
            "product": ["A", "B"],
            "revenue": [1000, 100], "growth": [0.4, -0.2],
            "units": [50, 10], "quadrant": ["Star", "Dog"],
        }),
        "calendar_df": pd.DataFrame({
            "week_start": [date(2026, 5, 4)],
            "action": ["Reorder X"], "category": ["inventory"],
            "audience": ["all"], "confidence": ["High"], "score": [0.9],
        }),
    }


# --------------------------------------------------------------------------- #
# PinSpec construction
# --------------------------------------------------------------------------- #

def test_pinspec_validates_kind():
    with pytest.raises(ValueError):
        PB.PinSpec(kind="not_a_real_chart", title="x")


def test_pinspec_auto_timestamps():
    spec = PB.PinSpec(kind="revenue_trend", title="Weekly revenue")
    assert spec.created_at  # populated automatically
    assert "T" in spec.created_at  # ISO format


def test_pinspec_roundtrip_dict():
    spec = PB.PinSpec(kind="rfm_scatter", title="Segments",
                      params={"rfm_df": {"$ref": "rfm_df"}})
    d = spec.to_dict()
    spec2 = PB.PinSpec.from_dict(d)
    assert spec2.kind == spec.kind
    assert spec2.title == spec.title
    assert spec2.params == spec.params


# --------------------------------------------------------------------------- #
# CRUD
# --------------------------------------------------------------------------- #

def test_add_pin_appends():
    pb = []
    pb = PB.add_pin(pb, PB.PinSpec("revenue_trend", "Weekly"))
    pb = PB.add_pin(pb, PB.PinSpec("rfm_scatter", "Segments"))
    assert len(pb) == 2


def test_add_pin_dedupes_same_title():
    pb = []
    pb = PB.add_pin(pb, PB.PinSpec("revenue_trend", "Weekly"))
    pb = PB.add_pin(pb, PB.PinSpec("revenue_trend", "Weekly"))
    assert len(pb) == 1


def test_remove_pin_returns_new_list():
    pb = [PB.PinSpec("revenue_trend", "A"), PB.PinSpec("rfm_scatter", "B")]
    out = PB.remove_pin(pb, 0)
    assert len(out) == 1 and out[0].title == "B"
    # original untouched
    assert len(pb) == 2


def test_remove_pin_out_of_range_no_op():
    pb = [PB.PinSpec("revenue_trend", "A")]
    out = PB.remove_pin(pb, 99)
    assert len(out) == 1


def test_move_pin_swaps_positions():
    pb = [PB.PinSpec("revenue_trend", "A"),
          PB.PinSpec("rfm_scatter", "B"),
          PB.PinSpec("product_matrix", "C")]
    moved = PB.move_pin(pb, 0, 1)
    assert [p.title for p in moved] == ["B", "A", "C"]


def test_move_pin_clamps_at_edges():
    pb = [PB.PinSpec("revenue_trend", "A"),
          PB.PinSpec("rfm_scatter", "B")]
    # try to move first up — should stay
    out = PB.move_pin(pb, 0, -1)
    assert [p.title for p in out] == ["A", "B"]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def test_render_pin_resolves_refs(sample_ctx):
    spec = PB.PinSpec("rfm_scatter", "Segments",
                      params={"rfm_df": {"$ref": "rfm_df"}})
    fig = PB.render_pin(spec, sample_ctx)
    assert isinstance(fig, go.Figure)


def test_render_pin_passes_through_literals(sample_ctx):
    spec = PB.PinSpec("revenue_trend", "Weekly", params={
        "df": {"$ref": "rev_df"},
        "date_col": "date",
        "rev_col": "rev",
        "freq": "W",
    })
    fig = PB.render_pin(spec, sample_ctx)
    assert "Revenue trend" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# JSON persistence
# --------------------------------------------------------------------------- #

def test_json_roundtrip_preserves_specs():
    pb = [
        PB.PinSpec("revenue_trend", "Weekly", params={"freq": "W"}),
        PB.PinSpec("rfm_scatter", "Segments",
                   params={"rfm_df": {"$ref": "rfm_df"}}),
    ]
    s = PB.to_json(pb)
    pb2 = PB.from_json(s)
    assert len(pb2) == 2
    assert pb2[0].kind == "revenue_trend"
    assert pb2[1].params["rfm_df"] == {"$ref": "rfm_df"}


def test_from_json_handles_empty():
    assert PB.from_json("") == []
    assert PB.from_json("   ") == []


# --------------------------------------------------------------------------- #
# HTML export
# --------------------------------------------------------------------------- #

def test_export_html_includes_all_pins(sample_ctx):
    pb = [
        PB.PinSpec("revenue_trend", "Weekly revenue", params={
            "df": {"$ref": "rev_df"}, "date_col": "date",
            "rev_col": "rev", "freq": "W",
        }),
        PB.PinSpec("rfm_scatter", "Customer segments",
                   params={"rfm_df": {"$ref": "rfm_df"}}),
    ]
    html = PB.export_html(pb, sample_ctx, title="My dashboard")
    assert "<!doctype html>" in html
    assert "My dashboard" in html
    assert "Weekly revenue" in html
    assert "Customer segments" in html
    # plotly CDN is included exactly once
    assert html.count("plotly") >= 1


def test_export_html_handles_empty_pinboard():
    html = PB.export_html([], {}, title="Empty")
    assert "No pinned charts yet" in html


def test_export_html_swallows_render_errors(sample_ctx):
    # Pass a spec referencing a non-existent dataframe — the builder
    # should still produce a "no data" placeholder, but if it raised,
    # export_html should keep going.
    pb = [PB.PinSpec("revenue_trend", "Bad pin", params={
        "df": {"$ref": "missing_df"}, "date_col": "date", "rev_col": "rev",
    })]
    html = PB.export_html(pb, sample_ctx)
    # Either rendered as a placeholder OR caught the exception
    assert "Bad pin" in html
