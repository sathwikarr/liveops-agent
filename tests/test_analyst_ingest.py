"""Tests for analyst/ingest.py — readers, type inference, classification."""
from __future__ import annotations

import io
import json
from textwrap import dedent

import pandas as pd
import pytest

from analyst import ingest as I


# --------------------------------------------------------------------------- #
# read_any
# --------------------------------------------------------------------------- #

def test_read_csv_basic():
    csv = "a,b,c\n1,2,3\n4,5,6\n"
    df = I.read_any(csv.encode("utf-8"))
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2


def test_read_csv_semicolon_delimiter():
    csv = "a;b;c\n1;2;3\n4;5;6\n"
    df = I.read_any(csv.encode("utf-8"))
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2


def test_read_csv_with_currency_strings():
    csv = "name,price\nA,$1,234.50\nB,$99.00\n"
    df = I.read_any(csv.encode("utf-8"))
    assert list(df.columns) == ["name", "price"]


def test_read_jsonl():
    payload = "\n".join(json.dumps({"x": i, "y": i * 2}) for i in range(3))
    df = I.read_any(payload.encode("utf-8"))
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 3


def test_read_json_array():
    payload = json.dumps([{"x": 1}, {"x": 2}])
    df = I.read_any(payload.encode("utf-8"))
    assert len(df) == 2


def test_read_json_nested_object():
    payload = json.dumps({"meta": "ignored", "rows": [{"a": 1}, {"a": 2}]})
    df = I.read_any(payload.encode("utf-8"))
    assert "a" in df.columns
    assert len(df) == 2


def test_read_handles_latin1_encoding():
    raw = "name,city\nrené,montréal\n".encode("latin-1")
    df = I.read_any(raw)
    assert "city" in df.columns


# --------------------------------------------------------------------------- #
# infer_schema
# --------------------------------------------------------------------------- #

def test_infer_datetime_column():
    df = pd.DataFrame({"order_date": ["2025-01-01", "2025-01-02", "2025-01-03"]})
    schema = I.infer_schema(df)
    col = schema.columns[0]
    assert col.inferred_type == "datetime"
    assert pd.api.types.is_datetime64_any_dtype(schema.coerced["order_date"])


def test_infer_currency_column():
    df = pd.DataFrame({"price": ["$1,234.50", "$99.00", "$45"]})
    schema = I.infer_schema(df)
    col = schema.columns[0]
    assert col.inferred_type == "currency"
    assert pd.api.types.is_numeric_dtype(schema.coerced["price"])
    assert schema.coerced["price"].iloc[0] == 1234.5


def test_infer_percent_column():
    df = pd.DataFrame({"margin": ["10%", "12.5%", "8%"]})
    schema = I.infer_schema(df)
    assert schema.columns[0].inferred_type == "percent"


def test_infer_boolean_column():
    df = pd.DataFrame({"is_active": ["true", "false", "true", "false"]})
    schema = I.infer_schema(df)
    assert schema.columns[0].inferred_type == "boolean"


def test_infer_id_column_high_uniqueness():
    df = pd.DataFrame({"order_id": [f"O-{i}" for i in range(20)]})
    schema = I.infer_schema(df)
    assert schema.columns[0].inferred_type == "id"


def test_infer_categorical_low_cardinality():
    df = pd.DataFrame({"region": ["East"] * 10 + ["West"] * 10 + ["North"] * 10})
    schema = I.infer_schema(df)
    assert schema.columns[0].inferred_type == "categorical"


def test_infer_text_high_cardinality_strings():
    # 20 distinct review-like strings
    df = pd.DataFrame({"review": [f"This is review number {i} with lots of text" for i in range(20)]})
    schema = I.infer_schema(df)
    assert schema.columns[0].inferred_type == "text"


def test_schema_helper_methods():
    df = pd.DataFrame({
        "ts": ["2025-01-01"] * 5,
        "rev": ["100"] * 5,
        "region": ["E"] * 5,
        "user_id": [f"U{i}" for i in range(5)],
    })
    schema = I.infer_schema(df)
    assert "ts" in schema.datetime_cols()
    assert "rev" in schema.numeric_cols()
    assert "region" in schema.categorical_cols()
    assert "user_id" in schema.id_cols()


# --------------------------------------------------------------------------- #
# classify_dataset
# --------------------------------------------------------------------------- #

def test_classify_sales_dataset():
    df = pd.DataFrame({
        "order_date": pd.date_range("2025-01-01", periods=10),
        "revenue": list(range(100, 110)),
        "product_id": ["P1"] * 10,
        "customer_id": [f"C{i}" for i in range(10)],
    })
    schema = I.infer_schema(df)
    kind = I.classify_dataset(df, schema)
    assert kind.kind == "sales"
    assert kind.confidence >= 0.8
    assert "date" in kind.role_map
    assert "amount" in kind.role_map


def test_classify_inventory_dataset():
    df = pd.DataFrame({
        "sku": ["A", "B", "C"],
        "inventory": [100, 200, 50],
        "product": ["X", "Y", "Z"],
    })
    schema = I.infer_schema(df)
    kind = I.classify_dataset(df, schema)
    assert kind.kind == "inventory"


def test_classify_generic_when_no_signals():
    df = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})
    schema = I.infer_schema(df)
    kind = I.classify_dataset(df, schema)
    assert kind.kind == "generic"


# --------------------------------------------------------------------------- #
# Top-level ingest()
# --------------------------------------------------------------------------- #

def test_ingest_end_to_end():
    csv = dedent("""\
        order_date,product_id,region,revenue,quantity
        2025-01-01,P1,East,$100.00,3
        2025-01-02,P2,West,$250.50,5
        2025-01-03,P1,East,$75.00,2
    """)
    result = I.ingest(csv.encode("utf-8"))
    assert result.n_rows == 3
    assert result.kind.kind == "sales"
    assert "revenue" in result.schema.numeric_cols()
    assert "order_date" in result.schema.datetime_cols()
    s = result.summary()
    assert s["rows"] == 3 and s["kind"]["kind"] == "sales"


def test_ingest_flags_high_null_columns():
    csv = "a,b\n1,\n2,\n3,\n4,\n"
    result = I.ingest(csv.encode("utf-8"))
    assert any("missing values" in i for i in result.issues)
