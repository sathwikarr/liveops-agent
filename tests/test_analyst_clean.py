"""Tests for analyst/clean.py — propose + apply with audit log."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyst import ingest as I
from analyst import clean as C


def test_propose_full_dupes_step():
    df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "y", "z"]})
    schema = I.infer_schema(df.astype(str))
    plan = C.propose(schema.coerced, schema)
    ids = [s.id for s in plan.steps]
    assert "drop_full_dupes" in ids


def test_apply_drops_dupes_and_logs_audit():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    schema = I.infer_schema(df.astype(str))
    plan = C.propose(schema.coerced, schema)
    cleaned, audit = C.apply(schema.coerced, plan)
    assert len(cleaned) == 2
    assert any(e.step_id == "drop_full_dupes" for e in audit)


def test_disabled_step_is_not_applied():
    df = pd.DataFrame({"a": [1, 1, 2]})
    schema = I.infer_schema(df.astype(str))
    plan = C.propose(schema.coerced, schema)
    for s in plan.steps:
        s.applied = False
    cleaned, audit = C.apply(schema.coerced, plan)
    assert len(cleaned) == 3
    assert audit == []


def test_propose_imputes_low_null_numeric():
    df = pd.DataFrame({"price": ["10", "20", "", "40", "50"], "name": ["a", "b", "c", "d", "e"]})
    schema = I.infer_schema(df)
    plan = C.propose(schema.coerced, schema)
    assert any(s.id.startswith("impute_") for s in plan.steps)


def test_apply_median_imputation_fills_nulls():
    df = pd.DataFrame({"price": ["10", "20", "", "40", "50"]})
    schema = I.infer_schema(df)
    plan = C.propose(schema.coerced, schema)
    cleaned, _ = C.apply(schema.coerced, plan)
    assert cleaned["price"].isna().sum() == 0


def test_strip_whitespace_step():
    df = pd.DataFrame({"region": [" East", "East ", " East ", "West"]})
    schema = I.infer_schema(df)
    plan = C.propose(schema.coerced, schema)
    assert any(s.id.startswith("strip_") for s in plan.steps)
    cleaned, _ = C.apply(schema.coerced, plan)
    assert set(cleaned["region"]) == {"East", "West"}


def test_drop_constant_column():
    df = pd.DataFrame({"x": [1, 2, 3], "const": ["A", "A", "A"]})
    schema = I.infer_schema(df.astype(str))
    plan = C.propose(schema.coerced, schema)
    assert any(s.id == "drop_const_const" for s in plan.steps)
    cleaned, _ = C.apply(schema.coerced, plan)
    assert "const" not in cleaned.columns


def test_clip_negative_amount_when_role_map_provided():
    df = pd.DataFrame({"order_date": pd.date_range("2025-01-01", periods=4),
                       "revenue": [-50, 100, 200, -10]})
    schema = I.infer_schema(df.astype(str))
    plan = C.propose(schema.coerced, schema, kind="sales",
                     role_map={"amount": "revenue", "date": "order_date"})
    assert any(s.id == "clip_neg_revenue" for s in plan.steps)
    cleaned, _ = C.apply(schema.coerced, plan)
    assert (cleaned["revenue"] >= 0).all()


def test_audit_records_row_counts():
    df = pd.DataFrame({"x": [1, 1, 2]})
    schema = I.infer_schema(df.astype(str))
    plan = C.propose(schema.coerced, schema)
    cleaned, audit = C.apply(schema.coerced, plan)
    if audit:
        e = audit[0]
        assert e.rows_before >= e.rows_after
        assert isinstance(e.notes, str)


def test_failing_step_does_not_crash():
    df = pd.DataFrame({"x": [1, 2]})
    schema = I.infer_schema(df.astype(str))
    bad = C.CleaningStep(
        id="explode", label="boom", description="raises",
        apply_fn=lambda d, p: (_ for _ in ()).throw(RuntimeError("nope")),
    )
    plan = C.CleaningPlan(steps=[bad])
    cleaned, audit = C.apply(df, plan)
    assert len(cleaned) == 2
    assert "FAILED" in audit[0].notes
