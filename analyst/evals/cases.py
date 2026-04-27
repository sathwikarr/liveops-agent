"""Eval corpus — 50+ questions covering all 10 agent tools.

Each tool gets ~5 cases mixing easy / paraphrased / adversarial phrasings.
A handful of multi-tool and "edge case" questions live at the bottom.

Tags are loose categories ("retention", "products", "schema") so the runner
can produce per-tag breakdowns. Adding tags is cheap; use them liberally.
"""
from __future__ import annotations

from typing import List

from analyst.evals.scorer import EvalCase


# --------------------------------------------------------------------------- #
# revenue_by_period (5)
# --------------------------------------------------------------------------- #

_REVENUE = [
    EvalCase(
        id="rev-001",
        question="Show me weekly revenue trend.",
        expected_tools=["revenue_by_period"],
        expected_args={"revenue_by_period": {"freq": "W"}},
        tags=["revenue", "easy"],
    ),
    EvalCase(
        id="rev-002",
        question="What does revenue look like over time?",
        expected_tools=["revenue_by_period"],
        expected_args={"revenue_by_period": {"freq": "W"}},
        tags=["revenue", "paraphrase"],
    ),
    EvalCase(
        id="rev-003",
        question="Plot daily revenue for the last quarter.",
        expected_tools=["revenue_by_period"],
        expected_args={"revenue_by_period": {"freq": "D"}},
        tags=["revenue", "args"],
    ),
    EvalCase(
        id="rev-004",
        question="Give me monthly sales by month.",
        expected_tools=["revenue_by_period"],
        expected_args={"revenue_by_period": {"freq": "M"}},
        tags=["revenue", "args"],
    ),
    EvalCase(
        id="rev-005",
        question="How are sales trending?",
        expected_tools=["revenue_by_period"],
        tags=["revenue", "vague"],
    ),
]


# --------------------------------------------------------------------------- #
# top_products (5)
# --------------------------------------------------------------------------- #

_TOP_PRODUCTS = [
    EvalCase(
        id="topp-001",
        question="What are the top 5 products by revenue?",
        expected_tools=["top_products"],
        expected_args={"top_products": {"n": 5}},
        forbidden_tools=["top_customers"],
        tags=["products", "easy", "args"],
    ),
    EvalCase(
        id="topp-002",
        question="Show me my best-selling SKUs.",
        expected_tools=["top_products"],
        forbidden_tools=["top_customers"],
        tags=["products", "paraphrase"],
    ),
    EvalCase(
        id="topp-003",
        question="Which 20 products bring in the most money?",
        expected_tools=["top_products"],
        expected_args={"top_products": {"n": 20}},
        tags=["products", "args"],
    ),
    EvalCase(
        id="topp-004",
        question="Top 3 products please.",
        expected_tools=["top_products"],
        expected_args={"top_products": {"n": 3}},
        tags=["products", "args", "terse"],
    ),
    EvalCase(
        id="topp-005",
        question="Best product overall?",
        expected_tools=["top_products"],
        tags=["products", "vague"],
    ),
]


# --------------------------------------------------------------------------- #
# top_customers (5)
# --------------------------------------------------------------------------- #

_TOP_CUSTOMERS = [
    EvalCase(
        id="topc-001",
        question="Show me my top 10 customers by revenue.",
        expected_tools=["top_customers"],
        expected_args={"top_customers": {"n": 10}},
        forbidden_tools=["top_products"],
        tags=["customers", "easy", "args"],
    ),
    EvalCase(
        id="topc-002",
        question="Who are my biggest spenders?",
        expected_tools=["top_customers"],
        forbidden_tools=["top_products"],
        tags=["customers", "paraphrase"],
    ),
    EvalCase(
        id="topc-003",
        question="List the top 25 customers.",
        expected_tools=["top_customers"],
        expected_args={"top_customers": {"n": 25}},
        tags=["customers", "args"],
    ),
    EvalCase(
        id="topc-004",
        question="Best customer this year?",
        expected_tools=["top_customers"],
        tags=["customers", "vague"],
    ),
    EvalCase(
        id="topc-005",
        question="Top 1 customer.",
        expected_tools=["top_customers"],
        expected_args={"top_customers": {"n": 1}},
        tags=["customers", "edge"],
    ),
]


# --------------------------------------------------------------------------- #
# segment_customers (5)
# --------------------------------------------------------------------------- #

_SEGMENT = [
    EvalCase(
        id="seg-001",
        question="Segment my customers using RFM.",
        expected_tools=["segment_customers"],
        tags=["segments", "easy"],
    ),
    EvalCase(
        id="seg-002",
        question="How many champions vs at-risk customers do I have?",
        expected_tools=["segment_customers"],
        tags=["segments", "paraphrase"],
    ),
    EvalCase(
        id="seg-003",
        question="Break down customers into loyalty segments.",
        expected_tools=["segment_customers"],
        tags=["segments", "paraphrase"],
    ),
    EvalCase(
        id="seg-004",
        question="What's the size of my Loyal segment?",
        expected_tools=["segment_customers"],
        tags=["segments", "specific"],
    ),
    EvalCase(
        id="seg-005",
        question="RFM analysis.",
        expected_tools=["segment_customers"],
        tags=["segments", "terse"],
    ),
]


# --------------------------------------------------------------------------- #
# product_quadrants (5)
# --------------------------------------------------------------------------- #

_QUADRANTS = [
    EvalCase(
        id="quad-001",
        question="Show me the BCG quadrants for my products.",
        expected_tools=["product_quadrants"],
        tags=["products", "quadrants", "easy"],
    ),
    EvalCase(
        id="quad-002",
        question="Which products are stars vs cash cows?",
        expected_tools=["product_quadrants"],
        tags=["products", "quadrants", "paraphrase"],
    ),
    EvalCase(
        id="quad-003",
        question="How many dog products do I have?",
        expected_tools=["product_quadrants"],
        tags=["products", "quadrants", "specific"],
    ),
    EvalCase(
        id="quad-004",
        question="Star product list?",
        expected_tools=["product_quadrants"],
        tags=["products", "quadrants", "terse"],
    ),
    EvalCase(
        id="quad-005",
        question="Question mark products?",
        expected_tools=["product_quadrants"],
        tags=["products", "quadrants", "terse"],
    ),
]


# --------------------------------------------------------------------------- #
# co_purchases (5)
# --------------------------------------------------------------------------- #

_COPURCHASE = [
    EvalCase(
        id="cop-001",
        question="What products are frequently bought together?",
        expected_tools=["co_purchases"],
        tags=["basket", "easy"],
    ),
    EvalCase(
        id="cop-002",
        question="Show me the top 15 co-purchase pairs.",
        expected_tools=["co_purchases"],
        expected_args={"co_purchases": {"n": 15}},
        tags=["basket", "args"],
    ),
    EvalCase(
        id="cop-003",
        question="Best cross-sell opportunities?",
        expected_tools=["co_purchases"],
        tags=["basket", "paraphrase"],
    ),
    EvalCase(
        id="cop-004",
        question="What should I bundle together?",
        expected_tools=["co_purchases"],
        tags=["basket", "paraphrase"],
    ),
    EvalCase(
        id="cop-005",
        question="Market basket analysis please.",
        expected_tools=["co_purchases"],
        tags=["basket", "terse"],
    ),
]


# --------------------------------------------------------------------------- #
# price_elasticity (5)
# --------------------------------------------------------------------------- #

_ELASTICITY = [
    EvalCase(
        id="ela-001",
        question="Compute price elasticity by product.",
        expected_tools=["price_elasticity"],
        tags=["elasticity", "easy"],
    ),
    EvalCase(
        id="ela-002",
        question="Which products are most price-sensitive?",
        expected_tools=["price_elasticity"],
        tags=["elasticity", "paraphrase"],
    ),
    EvalCase(
        id="ela-003",
        question="How much does demand drop when I raise the price?",
        expected_tools=["price_elasticity"],
        tags=["elasticity", "paraphrase"],
    ),
    EvalCase(
        id="ela-004",
        question="Elasticity scores by SKU.",
        expected_tools=["price_elasticity"],
        tags=["elasticity", "terse"],
    ),
    EvalCase(
        id="ela-005",
        question="Where is pricing power weakest?",
        expected_tools=["price_elasticity"],
        tags=["elasticity", "indirect"],
    ),
]


# --------------------------------------------------------------------------- #
# churn_risk (5)
# --------------------------------------------------------------------------- #

_CHURN = [
    EvalCase(
        id="chu-001",
        question="Which customers are most likely to churn?",
        expected_tools=["churn_risk"],
        tags=["retention", "easy"],
    ),
    EvalCase(
        id="chu-002",
        question="How many customers are inactive?",
        expected_tools=["churn_risk"],
        tags=["retention", "paraphrase"],
    ),
    EvalCase(
        id="chu-003",
        question="Show me customers at risk of leaving.",
        expected_tools=["churn_risk"],
        tags=["retention", "paraphrase"],
    ),
    EvalCase(
        id="chu-004",
        question="Distribution of retention risk?",
        expected_tools=["churn_risk"],
        tags=["retention", "specific"],
    ),
    EvalCase(
        id="chu-005",
        question="Churn analysis.",
        expected_tools=["churn_risk"],
        tags=["retention", "terse"],
    ),
]


# --------------------------------------------------------------------------- #
# cohort_retention (5)
# --------------------------------------------------------------------------- #

_COHORT = [
    EvalCase(
        id="coh-001",
        question="Show me monthly cohort retention.",
        expected_tools=["cohort_retention"],
        tags=["retention", "cohort", "easy"],
    ),
    EvalCase(
        id="coh-002",
        question="How does retention decay by signup month?",
        expected_tools=["cohort_retention"],
        tags=["retention", "cohort", "paraphrase"],
    ),
    EvalCase(
        id="coh-003",
        question="Cohort retention matrix?",
        expected_tools=["cohort_retention"],
        tags=["retention", "cohort", "terse"],
    ),
    EvalCase(
        id="coh-004",
        question="Build me a retention curve by cohort.",
        expected_tools=["cohort_retention"],
        tags=["retention", "cohort", "paraphrase"],
    ),
    EvalCase(
        id="coh-005",
        question="Monthly retention.",
        expected_tools=["cohort_retention"],
        tags=["retention", "terse"],
    ),
]


# --------------------------------------------------------------------------- #
# describe_columns (5)
# --------------------------------------------------------------------------- #

_DESCRIBE = [
    EvalCase(
        id="dsc-001",
        question="What columns are in my dataset?",
        expected_tools=["describe_columns"],
        tags=["schema", "easy"],
    ),
    EvalCase(
        id="dsc-002",
        question="Show me the schema.",
        expected_tools=["describe_columns"],
        tags=["schema", "terse"],
    ),
    EvalCase(
        id="dsc-003",
        question="What's in the data?",
        expected_tools=["describe_columns"],
        tags=["schema", "vague"],
    ),
    EvalCase(
        id="dsc-004",
        question="List dataset columns.",
        expected_tools=["describe_columns"],
        tags=["schema", "terse"],
    ),
    EvalCase(
        id="dsc-005",
        question="Describe the columns I have.",
        expected_tools=["describe_columns"],
        tags=["schema", "easy"],
    ),
]


# --------------------------------------------------------------------------- #
# Multi-tool / adversarial / edge cases (5)
# --------------------------------------------------------------------------- #

_MIXED = [
    EvalCase(
        id="mix-001",
        question="Which customers are churning, and what segments are they in?",
        expected_tools=["churn_risk", "segment_customers"],
        tags=["multi-tool", "retention"],
    ),
    EvalCase(
        id="mix-002",
        question="Show me the top products and their BCG quadrants.",
        expected_tools=["top_products", "product_quadrants"],
        tags=["multi-tool", "products"],
    ),
    EvalCase(
        id="mix-003",
        question="Hello, who are you?",
        # Heuristic agent falls back to describe_columns for unmatched questions.
        # That's the right behavior — we don't want it to invent a tool.
        expected_tools=["describe_columns"],
        tags=["adversarial", "off-topic"],
        notes="Off-topic question — fallback should kick in.",
    ),
    EvalCase(
        id="mix-004",
        question="What revenue did I do last week and which customers spent the most?",
        expected_tools=["revenue_by_period", "top_customers"],
        tags=["multi-tool", "revenue", "customers"],
    ),
    EvalCase(
        id="mix-005",
        question="Top 5 products and their elasticity?",
        expected_tools=["top_products", "price_elasticity"],
        expected_args={"top_products": {"n": 5}},
        tags=["multi-tool", "products"],
    ),
]


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #

ALL_CASES: List[EvalCase] = [
    *_REVENUE,
    *_TOP_PRODUCTS,
    *_TOP_CUSTOMERS,
    *_SEGMENT,
    *_QUADRANTS,
    *_COPURCHASE,
    *_ELASTICITY,
    *_CHURN,
    *_COHORT,
    *_DESCRIBE,
    *_MIXED,
]


def load_cases(*, tags: List[str] | None = None,
               ids: List[str] | None = None) -> List[EvalCase]:
    """Filter the full corpus.

    `tags`  — keep cases that have ANY of these tags (OR semantics)
    `ids`   — keep only these case ids
    """
    out = list(ALL_CASES)
    if ids:
        idset = set(ids)
        out = [c for c in out if c.id in idset]
    if tags:
        tagset = set(tags)
        out = [c for c in out if tagset & set(c.tags)]
    return out


__all__ = ["ALL_CASES", "load_cases"]
