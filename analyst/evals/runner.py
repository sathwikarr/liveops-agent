"""Eval runner — executes a corpus of cases through `analyst.agent.ask` and
aggregates per-tool / per-tag breakdowns.

Two entry points:

    run_eval(case, df, role_map, *, backend) -> CaseResult
    run_all(cases, df, role_map, *, backend, on_progress=...) -> EvalReport

The runner deliberately swallows exceptions raised by the agent and turns
them into a 0-score result with a `failed_reasons` entry — one buggy tool
shouldn't kill the whole run.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional

import pandas as pd

from analyst.agent import ask, AgentResult
from analyst.evals.scorer import (
    CaseResult, EvalCase, EvalReport, score_case,
)


def _default_role_map_for(df: pd.DataFrame) -> Dict[str, str]:
    """Best-effort role inference for a DataFrame the eval is being run on."""
    cols = {c.lower(): c for c in df.columns}
    role_map: Dict[str, str] = {}
    pairs = [
        ("customer", ["customer_id", "customer", "user_id", "user"]),
        ("date",     ["order_date", "date", "timestamp"]),
        ("amount",   ["revenue", "amount", "total", "sales"]),
        ("product",  ["product_id", "product", "sku"]),
        ("quantity", ["qty", "quantity", "units"]),
        ("price",    ["price", "unit_price"]),
        ("region",   ["city", "region", "country"]),
    ]
    for role, candidates in pairs:
        for cand in candidates:
            if cand in cols:
                role_map[role] = cols[cand]
                break
    return role_map


def run_eval(case: EvalCase, df: pd.DataFrame,
             role_map: Optional[Dict[str, str]] = None,
             *, backend: str = "heuristic") -> CaseResult:
    """Run a single case. Never raises — agent failures become a 0-score result."""
    rm = role_map or _default_role_map_for(df)
    try:
        result: AgentResult = ask(case.question, df, rm, backend=backend)
    except Exception as e:
        return CaseResult(
            case_id=case.id, question=case.question, backend=backend,
            planned_tools=[], planned_args={}, answer="",
            tool_match=0.0, args_match=0.0, no_forbidden=1.0,
            success_match=0.0, overall=0.0, passed=False,
            failed_reasons=[f"agent crashed: {type(e).__name__}: {e}"],
        )
    return score_case(
        case,
        plan_steps=result.plan.steps,
        observations=result.observations,
        answer=result.answer,
        backend=result.plan.backend,
    )


def _aggregate(cases: List[EvalCase], results: List[CaseResult],
               backend: str) -> EvalReport:
    """Roll per-case results up into an EvalReport, including per-tool +
    per-tag breakdowns."""
    n = len(results)
    if n == 0:
        return EvalReport(
            backend=backend, n_cases=0, n_passed=0, n_failed=0,
            mean_tool_match=0.0, mean_args_match=0.0,
            mean_no_forbidden=0.0, mean_success_match=0.0,
            mean_overall=0.0, pass_rate=0.0,
        )

    n_passed = sum(1 for r in results if r.passed)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0

    # per_tool: index cases by their (first) expected tool — then aggregate
    by_tool_pass: Dict[str, List[bool]] = defaultdict(list)
    by_tool_score: Dict[str, List[float]] = defaultdict(list)
    case_index = {c.id: c for c in cases}
    for r in results:
        c = case_index.get(r.case_id)
        if not c:
            continue
        for tool in c.expected_tools:
            by_tool_pass[tool].append(r.passed)
            by_tool_score[tool].append(r.overall)

    per_tool: Dict[str, Dict[str, float]] = {}
    for tool, passes in by_tool_pass.items():
        per_tool[tool] = {
            "n": float(len(passes)),
            "pass_rate": round(sum(passes) / len(passes), 4),
            "mean_overall": round(sum(by_tool_score[tool]) / len(passes), 4),
        }

    # per_tag
    by_tag_pass: Dict[str, List[bool]] = defaultdict(list)
    by_tag_score: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        c = case_index.get(r.case_id)
        if not c:
            continue
        for tag in c.tags:
            by_tag_pass[tag].append(r.passed)
            by_tag_score[tag].append(r.overall)
    per_tag: Dict[str, Dict[str, float]] = {}
    for tag, passes in by_tag_pass.items():
        per_tag[tag] = {
            "n": float(len(passes)),
            "pass_rate": round(sum(passes) / len(passes), 4),
            "mean_overall": round(sum(by_tag_score[tag]) / len(passes), 4),
        }

    return EvalReport(
        backend=backend,
        n_cases=n,
        n_passed=n_passed,
        n_failed=n - n_passed,
        mean_tool_match=mean([r.tool_match for r in results]),
        mean_args_match=mean([r.args_match for r in results]),
        mean_no_forbidden=mean([r.no_forbidden for r in results]),
        mean_success_match=mean([r.success_match for r in results]),
        mean_overall=mean([r.overall for r in results]),
        pass_rate=round(n_passed / n, 4),
        per_tool=per_tool,
        per_tag=per_tag,
        cases=results,
    )


def run_all(cases: List[EvalCase], df: pd.DataFrame,
            role_map: Optional[Dict[str, str]] = None,
            *, backend: str = "heuristic",
            on_progress: Optional[Callable[[int, int, CaseResult], None]] = None
            ) -> EvalReport:
    """Run every case in `cases` against `df`.

    `on_progress(idx, total, result)` is called after each case — useful for
    Streamlit progress bars.
    """
    rm = role_map or _default_role_map_for(df)
    results: List[CaseResult] = []
    for i, case in enumerate(cases, 1):
        r = run_eval(case, df, rm, backend=backend)
        results.append(r)
        if on_progress is not None:
            try:
                on_progress(i, len(cases), r)
            except Exception:
                pass
    return _aggregate(cases, results, backend=backend)


__all__ = ["run_eval", "run_all"]
