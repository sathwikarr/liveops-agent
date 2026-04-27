"""Eval data model + scoring logic.

Each `EvalCase` is a question paired with what we expect the agent to do.
Scoring is intentionally graded (0–1 per dimension) rather than binary so
we can see *how* the agent regressed, not just *that* it did.

Dimensions:
- tool_match     fraction of `expected_tools` present in the plan
- args_match     fraction of `expected_args` matching the plan's args
- no_forbidden   1.0 iff no `forbidden_tools` showed up
- success_match  fraction of `must_succeed` tools that ran without error

A case `passed` if tool_match >= 0.5, success_match >= 0.5, and no forbidden
tools were used. Tunable per-case via `pass_threshold`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Eval case definition
# --------------------------------------------------------------------------- #

@dataclass
class EvalCase:
    """A single eval point. `expected_tools` is what we want to see; everything
    else is optional but adds rigor."""
    id: str
    question: str
    expected_tools: List[str]
    forbidden_tools: List[str] = field(default_factory=list)
    expected_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    must_succeed: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    pass_threshold: float = 0.5
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("EvalCase needs a non-empty id")
        if not self.question or not self.question.strip():
            raise ValueError(f"Case {self.id}: question is empty")
        if not self.expected_tools:
            raise ValueError(f"Case {self.id}: expected_tools is empty")
        # If must_succeed isn't given, default it to expected_tools — we want
        # the tool to not just be picked but actually run.
        if not self.must_succeed:
            self.must_succeed = list(self.expected_tools)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Result + aggregate report
# --------------------------------------------------------------------------- #

@dataclass
class CaseResult:
    case_id: str
    question: str
    backend: str

    # Plan + observations from the agent run
    planned_tools: List[str]
    planned_args: Dict[str, Dict[str, Any]]
    answer: str

    # Scores in [0, 1]
    tool_match: float
    args_match: float
    no_forbidden: float
    success_match: float
    overall: float

    passed: bool
    failed_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    backend: str
    n_cases: int
    n_passed: int
    n_failed: int

    # Aggregates across all cases
    mean_tool_match: float
    mean_args_match: float
    mean_no_forbidden: float
    mean_success_match: float
    mean_overall: float

    pass_rate: float
    per_tool: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_tag: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cases: List[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["cases"] = [c.to_dict() for c in self.cases]
        return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _frac(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else 1.0


def _score_args(expected: Dict[str, Dict[str, Any]],
                planned: Dict[str, Dict[str, Any]]) -> float:
    """Each (tool, arg) pair in expected counts as one slot. A slot scores 1
    iff the tool was planned AND the arg matches. Missing slots score 0."""
    if not expected:
        return 1.0
    total = 0
    matched = 0
    for tool, args in expected.items():
        for arg, want in args.items():
            total += 1
            got_args = planned.get(tool, {})
            if arg in got_args and got_args[arg] == want:
                matched += 1
    return _frac(matched, total)


def score_case(case: EvalCase, *, plan_steps: List[Any],
               observations: List[Dict[str, Any]],
               answer: str, backend: str) -> CaseResult:
    """Grade one agent run against one case.

    `plan_steps` is `Plan.steps` (list of PlanStep). `observations` is the
    raw list returned by `agent._execute`.
    """
    planned_tools = [s.tool for s in plan_steps]
    planned_args = {s.tool: dict(s.args) for s in plan_steps}

    # 1. tool_match: fraction of expected tools that were planned
    expected_set = set(case.expected_tools)
    matched = expected_set & set(planned_tools)
    tool_match = _frac(len(matched), len(expected_set))

    # 2. args_match
    args_match = _score_args(case.expected_args, planned_args)

    # 3. no_forbidden — 1.0 if none of the forbidden tools showed up
    forbidden_hit = set(case.forbidden_tools) & set(planned_tools)
    no_forbidden = 0.0 if forbidden_hit else 1.0

    # 4. success_match: fraction of must_succeed tools that ran without error
    if case.must_succeed:
        ran_ok = 0
        for tool in case.must_succeed:
            for obs in observations:
                if obs["tool"] == tool:
                    res = obs.get("result", {})
                    if not (isinstance(res, dict) and "error" in res):
                        ran_ok += 1
                    break
        success_match = _frac(ran_ok, len(case.must_succeed))
    else:
        success_match = 1.0

    overall = (tool_match + args_match + no_forbidden + success_match) / 4.0

    failed_reasons: List[str] = []
    if tool_match < case.pass_threshold:
        failed_reasons.append(
            f"tool_match {tool_match:.2f} < {case.pass_threshold}: "
            f"expected {sorted(expected_set)}, planned {planned_tools}"
        )
    if forbidden_hit:
        failed_reasons.append(f"forbidden tools used: {sorted(forbidden_hit)}")
    if success_match < case.pass_threshold:
        failed_reasons.append(
            f"success_match {success_match:.2f} < {case.pass_threshold}"
        )

    passed = (
        tool_match >= case.pass_threshold
        and not forbidden_hit
        and success_match >= case.pass_threshold
    )

    return CaseResult(
        case_id=case.id,
        question=case.question,
        backend=backend,
        planned_tools=planned_tools,
        planned_args=planned_args,
        answer=answer,
        tool_match=round(tool_match, 4),
        args_match=round(args_match, 4),
        no_forbidden=round(no_forbidden, 4),
        success_match=round(success_match, 4),
        overall=round(overall, 4),
        passed=passed,
        failed_reasons=failed_reasons,
    )


__all__ = ["EvalCase", "CaseResult", "EvalReport", "score_case"]
