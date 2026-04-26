"""Tests for Stage 8 — calendar, multi-dataset join, competitor stub."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from analyst import calendar as Cal, join as J, competitor as Comp
from analyst.recommend import Recommendation


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #

def _rec(action, category, score):
    return Recommendation(
        action=action, evidence="…", confidence="High", impact_estimate="$",
        audience="all", category=category, bandit_arm=action.replace(" ", "_"),
        score=score,
    )


def test_calendar_places_top_score_first():
    recs = [_rec("Reorder P1", "inventory", 0.9),
            _rec("Promo on P2", "marketing", 0.8),
            _rec("Email cooling", "retention", 0.6)]
    cal = Cal.build_calendar(recs, weeks=2, start=date(2026, 4, 26))
    assert not cal.empty
    assert cal.iloc[0]["action"] == "Reorder P1"


def test_calendar_one_per_category_per_week():
    recs = [_rec("Reorder P1", "inventory", 0.9),
            _rec("Reorder P2", "inventory", 0.85),
            _rec("Reorder P3", "inventory", 0.8)]
    cal = Cal.build_calendar(recs, weeks=4, start=date(2026, 4, 26))
    # Should fill 3 different weeks, one per week, since same category
    assert cal["week_start"].nunique() == 3


def test_calendar_handles_empty():
    cal = Cal.build_calendar([], weeks=4)
    assert cal.empty


def test_calendar_week_starts_on_monday():
    recs = [_rec("X", "marketing", 0.5)]
    cal = Cal.build_calendar(recs, weeks=1, start=date(2026, 4, 26))  # Sunday
    ws = cal.iloc[0]["week_start"]
    assert ws.weekday() == 0  # Monday


# --------------------------------------------------------------------------- #
# Join
# --------------------------------------------------------------------------- #

def test_suggest_keys_finds_id_match():
    left = pd.DataFrame({"customer_id": ["A", "B", "C"], "rev": [1, 2, 3]})
    right = pd.DataFrame({"customer_id": ["A", "B"], "name": ["x", "y"]})
    sugg = J.suggest_keys(left, right)
    assert sugg
    assert sugg[0].left_col == "customer_id" and sugg[0].right_col == "customer_id"
    assert sugg[0].score > 0.7


def test_suggest_keys_finds_renamed_match_by_overlap():
    left = pd.DataFrame({"cust": ["A", "B", "C", "D"], "rev": [1, 2, 3, 4]})
    right = pd.DataFrame({"customer": ["A", "B", "C"], "name": ["x", "y", "z"]})
    sugg = J.suggest_keys(left, right)
    assert sugg
    # Even if names differ, overlap should still surface the pair
    assert {sugg[0].left_col, sugg[0].right_col} == {"cust", "customer"}


def test_auto_join_merges_on_best_key():
    left = pd.DataFrame({"customer_id": ["A", "B", "C"], "rev": [10, 20, 30]})
    right = pd.DataFrame({"customer_id": ["A", "B"], "ad_spend": [1, 2]})
    joined, key = J.auto_join(left, right)
    assert key is not None
    assert len(joined) == 2
    assert "rev" in joined.columns and "ad_spend" in joined.columns


def test_auto_join_returns_empty_when_no_match():
    left = pd.DataFrame({"a": [1, 2]})
    right = pd.DataFrame({"b": ["x", "y"]})
    joined, key = J.auto_join(left, right)
    assert joined.empty
    assert key is None


# --------------------------------------------------------------------------- #
# Competitor
# --------------------------------------------------------------------------- #

def test_competitor_stub_when_offline():
    df = Comp.lookup(["P1", "P2"], online=False)
    assert len(df) == 2
    assert (df["source"] == "stub").all()


def test_competitor_uses_fetch_fn_when_online():
    def fake_fetch(name):
        return {"competitor_price": 19.99, "trend": "down"}
    df = Comp.lookup(["P1"], online=True, fetch_fn=fake_fetch)
    assert set(df["signal"]) == {"competitor_price", "trend"}
    assert (df["source"] == "fetch").all()


def test_competitor_handles_fetch_error():
    def boom(name):
        raise RuntimeError("nope")
    df = Comp.lookup(["P1"], online=True, fetch_fn=boom)
    assert "error" in df["signal"].values
