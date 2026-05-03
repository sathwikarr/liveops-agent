"""Tests for analyst.agent — tool registry, heuristic planner, executor.
LLM backend is not exercised here (no API key in CI); covered indirectly
by the auto/heuristic fallback contract."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from analyst import agent as AG


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def retail_df():
    """Compact retail dataset that exercises every tool."""
    rows = []
    base = date(2026, 1, 1)
    customers = ["C1", "C2", "C3", "C4", "C5"] * 8
    products = ["P1", "P2", "P3", "P4"]
    order_id = 100
    for i, c in enumerate(customers):
        d = base + timedelta(days=i)
        # Each order has 2 products
        for j, p in enumerate(products[: 2 + (i % 2)]):
            rows.append({
                "order_id": order_id,
                "order_date": d.isoformat(),
                "customer_id": c,
                "product_id": p,
                "qty": 1 + (i + j) % 4,
                "price": 10.0 + (i + j) % 5,
                "revenue": (1 + (i + j) % 4) * (10.0 + (i + j) % 5),
            })
        order_id += 1
    df = pd.DataFrame(rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@pytest.fixture
def role_map():
    return {
        "customer": "customer_id", "date": "order_date", "amount": "revenue",
        "product": "product_id", "quantity": "qty", "price": "price",
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_tool_registry_has_expected_tools():
    expected = {
        "revenue_by_period", "top_products", "top_customers",
        "segment_customers", "product_quadrants", "co_purchases",
        "price_elasticity", "churn_risk", "cohort_retention",
        "describe_columns",
    }
    assert expected.issubset(AG.TOOLS.keys())


def test_every_tool_spec_has_callable_fn():
    for name, spec in AG.TOOLS.items():
        assert callable(spec.fn), f"{name} fn not callable"
        assert isinstance(spec.description, str) and spec.description
        assert isinstance(spec.params, dict)


# --------------------------------------------------------------------------- #
# Heuristic planner
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question,tool", [
    ("What does revenue look like over time?", "revenue_by_period"),
    ("Show me the top 5 products", "top_products"),
    ("Top customers by spend?", "top_customers"),
    ("Which customers are likely to churn?", "churn_risk"),
    ("How price-sensitive is each product?", "price_elasticity"),
    ("Show me co-purchases", "co_purchases"),
    ("RFM segments breakdown", "segment_customers"),
    ("BCG quadrants of products", "product_quadrants"),
    ("Cohort retention please", "cohort_retention"),
])
def test_heuristic_plan_routes_correctly(question, tool):
    plan = AG._heuristic_plan(question)
    tools = [s.tool for s in plan.steps]
    assert tool in tools
    assert plan.backend == "heuristic"


def test_heuristic_plan_falls_back_to_describe():
    plan = AG._heuristic_plan("complete gibberish about purple unicorns")
    assert plan.steps[0].tool == "describe_columns"


def test_heuristic_plan_extracts_n_from_topN():
    plan = AG._heuristic_plan("top 7 products please")
    step = next(s for s in plan.steps if s.tool == "top_products")
    assert step.args.get("n") == 7


# --------------------------------------------------------------------------- #
# Executor / individual tools
# --------------------------------------------------------------------------- #

def test_top_products_returns_ranked_list(retail_df, role_map):
    out = AG._t_top_products(retail_df, role_map, n=3)
    assert isinstance(out, list)
    assert len(out) == 3
    revenues = [r["revenue"] for r in out]
    assert revenues == sorted(revenues, reverse=True)


def test_top_customers_returns_ranked(retail_df, role_map):
    out = AG._t_top_customers(retail_df, role_map, n=3)
    assert len(out) == 3
    assert all("customer" in r and "revenue" in r for r in out)


def test_segment_customers(retail_df, role_map):
    out = AG._t_segment_customers(retail_df, role_map)
    assert isinstance(out, dict)
    assert sum(out.values()) > 0


def test_describe_columns_covers_every_column(retail_df, role_map):
    out = AG._t_describe_columns(retail_df, role_map)
    assert set(out.keys()) == set(retail_df.columns)
    for info in out.values():
        assert "dtype" in info and "null_pct" in info


def test_co_purchases_uses_order_id(retail_df, role_map):
    out = AG._t_co_purchases(retail_df, role_map)
    # Either pairs returned or an error explaining why none — never raises
    assert isinstance(out, (list, dict))


def test_revenue_by_period_returns_records(retail_df, role_map):
    out = AG._t_revenue_by_period(retail_df, role_map, freq="W")
    assert isinstance(out, list)


def test_tool_returns_error_on_missing_role(retail_df):
    out = AG._t_top_customers(retail_df, role_map={}, n=5)
    assert "error" in out


# --------------------------------------------------------------------------- #
# End-to-end ask()
# --------------------------------------------------------------------------- #

def test_ask_heuristic_top_products(retail_df, role_map):
    res = AG.ask("Show me the top 3 products", retail_df, role_map,
                 backend="heuristic")
    assert isinstance(res, AG.AgentResult)
    assert res.plan.backend == "heuristic"
    assert any(o["tool"] == "top_products" for o in res.observations)
    # 3 rows requested
    obs = next(o for o in res.observations if o["tool"] == "top_products")
    assert len(obs["result"]) == 3


def test_ask_returns_answer_text(retail_df, role_map):
    res = AG.ask("Revenue trend over time?", retail_df, role_map,
                 backend="heuristic")
    assert isinstance(res.answer, str)
    assert len(res.answer) > 0


def test_ask_handles_missing_role_gracefully(retail_df):
    res = AG.ask("Top customers", retail_df, role_map={}, backend="heuristic")
    # Tool ran, returned an error in result, ask still produced an answer
    obs = next(o for o in res.observations if o["tool"] == "top_customers")
    assert "error" in obs["result"]
    assert isinstance(res.answer, str) and res.answer


def test_ask_serializes_to_dict(retail_df, role_map):
    res = AG.ask("Show segments", retail_df, role_map, backend="heuristic")
    d = res.to_dict()
    assert "answer" in d and "plan" in d and "observations" in d
    assert d["plan"]["backend"] == "heuristic"


# --------------------------------------------------------------------------- #
# Plan `why` strings must be user-facing prose, not raw regex
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, expect_in_why", [
    ("top 5 products by revenue",  ('rank SKUs by total revenue', 'n=5')),
    ("who is about to churn?",     ('churn risk',)),
    ("show me cohort retention",   ('signup-cohort retention curve',)),
    ("rfm segments",               ('RFM segments',)),
    ("product quadrants",          ('BCG quadrant',)),
    ("are my customers price sensitive?", ('demand', 'price')),
    ("weekly revenue trend",       ('aggregate revenue by week', 'weekly')),
    ("monthly revenue",            ('aggregate revenue by month', 'monthly')),
    ("best 3 customers",           ('rank customers', 'n=3')),
    ("frequently bought together", ('co-purchase',)),
    ("describe columns",           ('list the columns',)),
])
def test_heuristic_why_is_user_prose(retail_df, role_map, question, expect_in_why):
    """The plan step's `why` field is rendered directly in the workbench UI.
    It MUST read as English to a non-developer — no regex literals, no
    internal jargon. This regression test catches a future refactor that
    accidentally surfaces /\\b(?:foo|bar)\\b/ to users."""
    plan = AG._heuristic_plan(question)
    whys = " | ".join(s.why for s in plan.steps)
    # Negative assertions — none of the regex artefacts should leak through.
    for forbidden in ("matched /", "\\b", "\\w", "(?:", "(?P<"):
        assert forbidden not in whys, f"regex artefact {forbidden!r} leaked into why: {whys!r}"
    # Positive — the friendly intent fragments must be present somewhere.
    for needle in expect_in_why:
        assert needle.lower() in whys.lower(), \
            f"expected {needle!r} in any step's why for {question!r}, got {whys!r}"


def test_heuristic_garbage_query_helpful_fallback():
    """Garbage queries route to describe_columns with a helpful 'try
    rephrasing' message, not the old terse 'fallback' string."""
    plan = AG._heuristic_plan("asdfghjk qwertyu")
    assert len(plan.steps) == 1
    why = plan.steps[0].why
    assert "describing the dataset's columns" in why
    assert "try rephrasing" in why
    # Must mention at least one concrete keyword to nudge the user
    assert any(kw in why.lower() for kw in ("revenue", "churn", "products"))


def test_executor_swallows_tool_exceptions():
    """If a tool raises, the executor must capture it as an error observation
    rather than letting the exception bubble up."""
    bad_spec = AG.ToolSpec("bad", "always raises", {},
                           lambda df, role_map, **kw: (_ for _ in ()).throw(RuntimeError("kaboom")))
    AG.TOOLS["__bad__"] = bad_spec
    try:
        plan = AG.Plan(steps=[AG.PlanStep(tool="__bad__", args={}, why="test")],
                       reasoning="test", backend="heuristic")
        obs = AG._execute(plan, pd.DataFrame({"x": [1]}), {})
        assert "error" in obs[0]["result"]
        assert "kaboom" in obs[0]["result"]["error"]
    finally:
        del AG.TOOLS["__bad__"]


def test_executor_filters_invalid_kwargs():
    """If the LLM hallucinates a param the tool doesn't accept, the
    executor retries with only valid kwargs."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    plan = AG.Plan(steps=[AG.PlanStep(tool="describe_columns",
                                      args={"bogus_param": 99},
                                      why="test")],
                   backend="heuristic")
    obs = AG._execute(plan, df, {})
    # Should NOT have an error — the bogus param was stripped
    assert "error" not in obs[0]["result"]
    assert "x" in obs[0]["result"]


# --------------------------------------------------------------------------- #
# Format observation rendering
# --------------------------------------------------------------------------- #

def test_format_observation_dict_result():
    obs = {"tool": "segment_customers",
           "result": {"Champions": 5, "Lost": 3}}
    out = AG._format_observation(obs)
    assert "segment_customers" in out and "Champions" in out


def test_format_observation_error_result():
    obs = {"tool": "x", "result": {"error": "missing roles"}}
    out = AG._format_observation(obs)
    assert "missing roles" in out


def test_format_observation_list_of_dicts():
    obs = {"tool": "top_products",
           "result": [{"product": "A", "revenue": 100.0},
                      {"product": "B", "revenue": 50.0}]}
    out = AG._format_observation(obs)
    assert "top_products" in out and "A" in out
