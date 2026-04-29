"""Held-out eval corpus — questions the heuristic regex was NEVER tuned on.

The original 55-case corpus and the heuristic in `analyst.agent._HEURISTIC_PATTERNS`
were written by the same person at the same time, so the pinned 100% pass rate
is closer to "the regex still matches itself" than "the agent generalizes."

This module exists to give an honest signal. Every case below intentionally
avoids the keywords baked into the heuristic regex (no "revenue", "top
products", "churn", "co-purchase", "elasticity", "cohort", etc.) and instead
uses paraphrases a real user might type. We expect the heuristic to fail
several of these — that failure mode is the whole point of the harness.

When you pin a baseline from this corpus, save it to:

    tests/fixtures/eval_holdout_baseline_heuristic.json
    tests/fixtures/eval_holdout_baseline_llm.json   (when GEMINI_API_KEY is set)
"""
from __future__ import annotations

from typing import List

from analyst.evals.scorer import EvalCase


HOLDOUT_CASES: List[EvalCase] = [
    # --- revenue_by_period -------------------------------------------------- #
    EvalCase(
        id="ho-rev-001",
        question="How much money are we pulling in each week?",
        expected_tools=["revenue_by_period"],
        expected_args={"revenue_by_period": {"freq": "W"}},
        tags=["holdout", "revenue", "paraphrase"],
        notes="Avoids 'revenue', 'sales', 'trend' — uses 'money pulling in'.",
    ),
    EvalCase(
        id="ho-rev-002",
        question="What's the income picture month by month?",
        expected_tools=["revenue_by_period"],
        expected_args={"revenue_by_period": {"freq": "M"}},
        tags=["holdout", "revenue", "paraphrase", "args"],
        notes="'Income' is unmatched by the heuristic; 'month by month' should still set freq=M.",
    ),

    # --- top_products ------------------------------------------------------- #
    EvalCase(
        id="ho-topp-001",
        question="Which items are flying off the shelves?",
        expected_tools=["top_products"],
        forbidden_tools=["top_customers"],
        tags=["holdout", "products", "idiom"],
        notes="No 'top'/'best'/'products'/'skus' keyword.",
    ),
    EvalCase(
        id="ho-topp-002",
        question="Show me my heaviest hitters in the catalog.",
        expected_tools=["top_products"],
        forbidden_tools=["top_customers"],
        tags=["holdout", "products", "idiom"],
    ),

    # --- top_customers ------------------------------------------------------ #
    EvalCase(
        id="ho-topc-001",
        question="Who's on our VIP list?",
        expected_tools=["top_customers"],
        forbidden_tools=["top_products"],
        tags=["holdout", "customers", "idiom"],
    ),
    EvalCase(
        id="ho-topc-002",
        question="Identify my whales.",
        expected_tools=["top_customers"],
        forbidden_tools=["top_products"],
        tags=["holdout", "customers", "idiom"],
        notes="Gaming/finance slang for high-spend customers.",
    ),

    # --- segment_customers -------------------------------------------------- #
    EvalCase(
        id="ho-seg-001",
        question="Bucket my buyers into groups based on behavior.",
        expected_tools=["segment_customers"],
        tags=["holdout", "segments", "paraphrase"],
        notes="No 'segment'/'rfm'/'loyal' keyword.",
    ),
    EvalCase(
        id="ho-seg-002",
        question="Cluster the customer base.",
        expected_tools=["segment_customers"],
        tags=["holdout", "segments", "paraphrase"],
    ),

    # --- product_quadrants -------------------------------------------------- #
    EvalCase(
        id="ho-quad-001",
        question="Which products are winners and which are losers?",
        expected_tools=["product_quadrants"],
        tags=["holdout", "quadrants", "paraphrase"],
        notes="No 'quadrant'/'stars'/'cash cow'/'BCG' vocabulary.",
    ),

    # --- co_purchases ------------------------------------------------------- #
    EvalCase(
        id="ho-co-001",
        question="What products get bought together?",
        expected_tools=["co_purchases"],
        tags=["holdout", "basket", "paraphrase"],
        notes="No 'co-purchase'/'basket'/'frequently bought'/'bundle' keyword.",
    ),

    # --- price_elasticity --------------------------------------------------- #
    EvalCase(
        id="ho-elas-001",
        question="If I bump prices 10% what happens to volume?",
        expected_tools=["price_elasticity"],
        tags=["holdout", "pricing", "paraphrase"],
        notes="No 'elasticity'/'price-sensitive' keyword; 'bump prices' instead.",
    ),

    # --- churn_risk --------------------------------------------------------- #
    EvalCase(
        id="ho-churn-001",
        question="Who's gone quiet on us recently?",
        expected_tools=["churn_risk"],
        tags=["holdout", "retention", "idiom"],
        notes="No 'churn'/'leaving'/'inactive' keyword; 'gone quiet' instead.",
    ),

    # --- cohort_retention --------------------------------------------------- #
    EvalCase(
        id="ho-coh-001",
        question="Of the folks who signed up in January, how many are still around?",
        expected_tools=["cohort_retention"],
        tags=["holdout", "retention", "paraphrase"],
        notes="No 'cohort'/'retention' keyword.",
    ),

    # --- describe_columns --------------------------------------------------- #
    EvalCase(
        id="ho-desc-001",
        question="What fields are available in this dataset?",
        expected_tools=["describe_columns"],
        tags=["holdout", "schema", "paraphrase"],
        notes="'Fields' instead of 'columns'/'schema'.",
    ),
]


def load_holdout_cases(*, tags: List[str] | None = None,
                       ids: List[str] | None = None) -> List[EvalCase]:
    """Same filter semantics as `load_cases`, scoped to the held-out set."""
    out = list(HOLDOUT_CASES)
    if ids:
        idset = set(ids)
        out = [c for c in out if c.id in idset]
    if tags:
        tagset = set(tags)
        out = [c for c in out if tagset & set(c.tags)]
    return out


__all__ = ["HOLDOUT_CASES", "load_holdout_cases"]
