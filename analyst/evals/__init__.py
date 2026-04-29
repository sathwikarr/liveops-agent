"""Eval harness for the LLM agent — runs a corpus of questions through
`analyst.agent.ask`, scores each against an expected plan, and aggregates
pass/fail metrics so we can spot regressions.

Public surface:

    from analyst.evals import (
        EvalCase, CaseResult, EvalReport,
        run_eval, run_all, load_cases,
    )

Run from the CLI with:

    python -m analyst.evals --backend heuristic
"""
from __future__ import annotations

from analyst.evals.scorer import (
    EvalCase, CaseResult, EvalReport, score_case,
)
from analyst.evals.runner import run_eval, run_all
from analyst.evals.cases import load_cases, ALL_CASES
from analyst.evals.holdout_cases import HOLDOUT_CASES, load_holdout_cases

__all__ = [
    "EvalCase", "CaseResult", "EvalReport", "score_case",
    "run_eval", "run_all",
    "load_cases", "ALL_CASES",
    "load_holdout_cases", "HOLDOUT_CASES",
]
