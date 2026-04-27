"""Tests for the eval harness — EvalCase validation, scorer math, runner
behavior, and an end-to-end check that the bundled corpus passes its own
pinned baseline on the heuristic backend.

Why bother eval-testing the eval harness? Because if scoring drifts silently,
every "pass rate is 100%" claim downstream becomes a lie. Same reason CI
tests its own assertions before tests pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analyst.agent import Plan, PlanStep
from analyst.evals import (
    ALL_CASES, EvalCase, CaseResult, EvalReport, load_cases,
    run_eval, run_all, score_case,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "analyst" / "sample_data" / "retail_orders.csv"


@pytest.fixture(scope="module")
def sample_df():
    if not SAMPLE_CSV.exists():
        pytest.skip(f"Sample data not present at {SAMPLE_CSV}")
    return pd.read_csv(SAMPLE_CSV)


# --------------------------------------------------------------------------- #
# EvalCase validation
# --------------------------------------------------------------------------- #

def test_eval_case_requires_id():
    with pytest.raises(ValueError):
        EvalCase(id="", question="x", expected_tools=["top_products"])


def test_eval_case_requires_question():
    with pytest.raises(ValueError):
        EvalCase(id="x", question="", expected_tools=["top_products"])


def test_eval_case_requires_expected_tools():
    with pytest.raises(ValueError):
        EvalCase(id="x", question="q", expected_tools=[])


def test_eval_case_must_succeed_defaults_to_expected_tools():
    c = EvalCase(id="x", question="q", expected_tools=["top_products"])
    assert c.must_succeed == ["top_products"]


# --------------------------------------------------------------------------- #
# Scorer
# --------------------------------------------------------------------------- #

def _plan(*pairs):
    """Build a Plan from a list of (tool_name, args) tuples."""
    return [PlanStep(tool=t, args=dict(a)) for t, a in pairs]


def test_score_perfect_match():
    case = EvalCase(id="t", question="q", expected_tools=["top_products"],
                    expected_args={"top_products": {"n": 5}})
    obs = [{"tool": "top_products", "args": {"n": 5}, "result": [{"x": 1}]}]
    r = score_case(case, plan_steps=_plan(("top_products", {"n": 5})),
                   observations=obs, answer="ok", backend="heuristic")
    assert r.passed
    assert r.tool_match == 1.0
    assert r.args_match == 1.0
    assert r.no_forbidden == 1.0
    assert r.success_match == 1.0
    assert r.overall == 1.0


def test_score_wrong_tool_fails():
    case = EvalCase(id="t", question="q", expected_tools=["top_products"])
    obs = [{"tool": "describe_columns", "args": {}, "result": {}}]
    r = score_case(case, plan_steps=_plan(("describe_columns", {})),
                   observations=obs, answer="ok", backend="heuristic")
    assert not r.passed
    assert r.tool_match == 0.0
    assert any("tool_match" in reason for reason in r.failed_reasons)


def test_score_partial_tool_match_passes_at_threshold():
    case = EvalCase(id="t", question="q",
                    expected_tools=["top_products", "churn_risk"],
                    must_succeed=["top_products"])
    # Plan only includes one of the two — 0.5 match exactly hits threshold
    obs = [{"tool": "top_products", "args": {}, "result": [{"x": 1}]}]
    r = score_case(case, plan_steps=_plan(("top_products", {})),
                   observations=obs, answer="ok", backend="heuristic")
    assert r.tool_match == 0.5
    assert r.passed   # threshold is >= 0.5


def test_score_forbidden_tool_fails_even_if_correct_one_present():
    case = EvalCase(id="t", question="q",
                    expected_tools=["top_products"],
                    forbidden_tools=["top_customers"])
    obs = [
        {"tool": "top_products", "args": {}, "result": [{"x": 1}]},
        {"tool": "top_customers", "args": {}, "result": [{"x": 2}]},
    ]
    r = score_case(case, plan_steps=_plan(("top_products", {}),
                                           ("top_customers", {})),
                   observations=obs, answer="ok", backend="heuristic")
    assert r.no_forbidden == 0.0
    assert not r.passed
    assert any("forbidden" in reason for reason in r.failed_reasons)


def test_score_args_mismatch_lowers_args_match():
    case = EvalCase(id="t", question="q", expected_tools=["top_products"],
                    expected_args={"top_products": {"n": 5}})
    obs = [{"tool": "top_products", "args": {"n": 10}, "result": [{}]}]
    r = score_case(case, plan_steps=_plan(("top_products", {"n": 10})),
                   observations=obs, answer="ok", backend="heuristic")
    assert r.tool_match == 1.0       # right tool
    assert r.args_match == 0.0       # wrong arg value
    # Still passes — args_match doesn't gate pass/fail


def test_score_must_succeed_failure_blocks_pass():
    case = EvalCase(id="t", question="q", expected_tools=["top_products"])
    # Tool was planned but errored at execution
    obs = [{"tool": "top_products", "args": {}, "result": {"error": "boom"}}]
    r = score_case(case, plan_steps=_plan(("top_products", {})),
                   observations=obs, answer="ok", backend="heuristic")
    assert r.tool_match == 1.0
    assert r.success_match == 0.0
    assert not r.passed


# --------------------------------------------------------------------------- #
# Runner — agent crash isolation
# --------------------------------------------------------------------------- #

def test_run_eval_handles_agent_crash(monkeypatch, sample_df):
    case = EvalCase(id="crash", question="will it blend?",
                    expected_tools=["top_products"])

    def boom(*a, **kw):
        raise RuntimeError("simulated agent crash")

    import analyst.evals.runner as rn
    monkeypatch.setattr(rn, "ask", boom)

    r = rn.run_eval(case, sample_df, backend="heuristic")
    assert isinstance(r, CaseResult)
    assert not r.passed
    assert any("crashed" in reason for reason in r.failed_reasons)


def test_run_eval_default_role_map(sample_df):
    """Sanity check: default role inference picks the right columns."""
    from analyst.evals.runner import _default_role_map_for
    rm = _default_role_map_for(sample_df)
    assert rm.get("customer") == "customer_id"
    assert rm.get("date") == "order_date"
    assert rm.get("amount") == "revenue"
    assert rm.get("product") == "product_id"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def test_run_all_aggregates_per_tool_and_per_tag(sample_df):
    cases = load_cases(tags=["easy"])
    assert len(cases) >= 5
    report = run_all(cases, sample_df, backend="heuristic")
    assert isinstance(report, EvalReport)
    assert report.n_cases == len(cases)
    assert 0.0 <= report.pass_rate <= 1.0
    # per-tool breakdown should cover at least the tools mentioned in the cases
    expected_tools = {t for c in cases for t in c.expected_tools}
    assert set(report.per_tool.keys()) >= expected_tools
    # 'easy' tag should appear in per-tag
    assert "easy" in report.per_tag


def test_run_all_progress_callback_invoked(sample_df):
    cases = load_cases(tags=["terse"])[:3]
    seen = []
    run_all(cases, sample_df, backend="heuristic",
            on_progress=lambda i, n, r: seen.append((i, n, r.case_id)))
    assert len(seen) == len(cases)
    assert [s[0] for s in seen] == list(range(1, len(cases) + 1))


def test_run_all_progress_callback_failure_doesnt_break_run(sample_df):
    cases = load_cases(tags=["terse"])[:2]
    def boom(i, n, r):
        raise ValueError("don't kill the run")
    report = run_all(cases, sample_df, backend="heuristic", on_progress=boom)
    assert report.n_cases == len(cases)


# --------------------------------------------------------------------------- #
# Corpus integrity
# --------------------------------------------------------------------------- #

def test_corpus_size():
    assert len(ALL_CASES) >= 50, "Eval corpus should have at least 50 cases"


def test_corpus_covers_every_tool():
    from analyst.agent import TOOLS
    expected = {t for c in ALL_CASES for t in c.expected_tools}
    # Every registered tool should appear as an expected_tool somewhere
    for tool in TOOLS:
        assert tool in expected, f"Tool {tool!r} is not exercised by any case"


def test_corpus_ids_unique():
    ids = [c.id for c in ALL_CASES]
    assert len(ids) == len(set(ids)), "Duplicate case ids in corpus"


def test_corpus_only_uses_real_tools():
    from analyst.agent import TOOLS
    real = set(TOOLS.keys())
    for c in ALL_CASES:
        for t in c.expected_tools:
            assert t in real, f"Case {c.id}: unknown tool {t!r}"
        for t in c.forbidden_tools:
            assert t in real, f"Case {c.id}: unknown forbidden tool {t!r}"


def test_load_cases_filter_by_tag():
    easy = load_cases(tags=["easy"])
    assert all("easy" in c.tags for c in easy)


def test_load_cases_filter_by_id():
    out = load_cases(ids=["rev-001", "topp-001"])
    assert {c.id for c in out} == {"rev-001", "topp-001"}


# --------------------------------------------------------------------------- #
# End-to-end — heuristic backend on bundled data must beat baseline
# --------------------------------------------------------------------------- #

def test_heuristic_backend_meets_baseline(sample_df):
    """The pinned baseline says heuristic backend should hit ~100% on the
    bundled retail dataset. If we drift below 0.95, something regressed."""
    report = run_all(ALL_CASES, sample_df, backend="heuristic")
    assert report.pass_rate >= 0.95, (
        f"Heuristic pass rate dropped to {report.pass_rate:.2%}. "
        f"Failures: {[r.case_id for r in report.cases if not r.passed]}"
    )


def test_baseline_file_is_consistent_with_corpus():
    """The pinned baseline should match the current corpus size — if a case
    is added, the baseline needs to be re-generated."""
    baseline_path = REPO_ROOT / "tests" / "fixtures" / "eval_baseline.json"
    if not baseline_path.exists():
        pytest.skip("No baseline pinned yet")
    base = json.loads(baseline_path.read_text())
    assert base["n_cases"] == len(ALL_CASES), (
        f"Baseline has {base['n_cases']} cases but corpus has "
        f"{len(ALL_CASES)}. Re-run `python -m analyst.evals --json "
        f"tests/fixtures/eval_baseline.json` to refresh."
    )
