"""LLM agent loop — registry of analyst tools + a planner that picks them.

Why this exists: most "AI dashboards" are LLM wrappers around static charts.
This module gives the analyst a real tool-using loop:

    user question  →  planner (LLM or heuristic) picks tools
                  →  tools run on the live DataFrame
                  →  observations fed back to a synthesizer (LLM or template)
                  →  natural-language answer + the trace of tools used

The planner has two backends:
1. LLM backend (Gemini) — used when GEMINI_API_KEY is configured
2. Heuristic backend — keyword routing; deterministic, offline-safe

Both produce the same `Plan` shape, so the executor doesn't care which
ran. That makes the demo work without any API key, and lets unit tests
exercise the loop without mocking an LLM.

Public API:
- TOOLS: dict[name, ToolSpec]
- Plan: dataclass with steps + reasoning
- AgentResult: dataclass with answer, plan, observations
- ask(question, df, role_map=None, *, backend="auto") -> AgentResult
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from analyst import analysis as A
from analyst import predict as P


# --------------------------------------------------------------------------- #
# Tool registry — every tool is a (callable, schema, doc) tuple
# --------------------------------------------------------------------------- #

@dataclass
class ToolSpec:
    name: str
    description: str
    params: Dict[str, str]   # param_name -> human description
    fn: Callable[..., Any]   # signature: (df, role_map, **params) -> Any

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "params": self.params}


def _t_revenue_by_period(df, role_map, *, freq: str = "W"):
    if "date" not in role_map or "amount" not in role_map:
        return {"error": "Need date + amount roles"}
    return A.revenue_trend(df, role_map=role_map, freq=freq).to_dict("records")


def _t_top_products(df, role_map, *, n: int = 10):
    if "product" not in role_map or "amount" not in role_map:
        return {"error": "Need product + amount roles"}
    g = (df.groupby(role_map["product"])[role_map["amount"]]
           .sum().sort_values(ascending=False).head(int(n)))
    return [{"product": p, "revenue": float(v)} for p, v in g.items()]


def _t_top_customers(df, role_map, *, n: int = 10):
    if "customer" not in role_map or "amount" not in role_map:
        return {"error": "Need customer + amount roles"}
    g = (df.groupby(role_map["customer"])[role_map["amount"]]
           .sum().sort_values(ascending=False).head(int(n)))
    return [{"customer": c, "revenue": float(v)} for c, v in g.items()]


def _t_segment_customers(df, role_map):
    rfm = A.rfm(df, role_map=role_map)
    if rfm.empty:
        return {"error": "RFM needs customer + date + amount"}
    return rfm["segment"].value_counts().to_dict()


def _t_product_quadrants(df, role_map):
    pm = A.product_matrix(df, role_map=role_map)
    if pm.empty:
        return {"error": "product_matrix needs product + amount + date"}
    return {q: int(c) for q, c in pm["quadrant"].value_counts().items()}


def _t_co_purchases(df, role_map, *, basket_key: Optional[str] = None, n: int = 10):
    bk = basket_key
    if not bk and "order_id" in df.columns:
        bk = "order_id"
    basket = A.market_basket(df, role_map=role_map, basket_key=bk, top_n=int(n))
    if basket.empty:
        return {"error": "Not enough baskets to compute co-purchases"}
    return basket.head(int(n)).to_dict("records")


def _t_price_elasticity(df, role_map):
    el = A.elasticity(df, role_map=role_map)
    if el.empty:
        return {"error": "Need product + price + quantity to estimate elasticity"}
    return el.head(20).to_dict("records")


def _t_churn_risk(df, role_map, *, days: int = 60):
    ch = P.churn_scores(df, role_map=role_map, churn_window_days=int(days))
    if ch.empty:
        return {"error": "Need customer + date to score churn"}
    return ch["risk"].value_counts().to_dict()


def _t_cohort_retention(df, role_map):
    cr = A.cohort_retention(df, role_map=role_map)
    if cr.empty:
        return {"error": "Need customer + date for cohort analysis"}
    return cr.round(3).to_dict()


def _t_describe_columns(df, role_map):
    out = {}
    for col in df.columns:
        s = df[col]
        info = {"dtype": str(s.dtype), "null_pct": float(s.isna().mean().round(3))}
        if pd.api.types.is_numeric_dtype(s):
            info.update({"mean": float(s.mean()) if not s.empty else None,
                         "min": float(s.min()) if not s.empty else None,
                         "max": float(s.max()) if not s.empty else None})
        else:
            info["nunique"] = int(s.nunique())
        out[col] = info
    return out


TOOLS: Dict[str, ToolSpec] = {
    "revenue_by_period": ToolSpec(
        "revenue_by_period",
        "Total revenue aggregated by period (D, W, M).",
        {"freq": "Resampling frequency: D=daily, W=weekly, M=monthly. Default W."},
        _t_revenue_by_period,
    ),
    "top_products": ToolSpec(
        "top_products",
        "Top products by total revenue.",
        {"n": "How many products to return (default 10)."},
        _t_top_products,
    ),
    "top_customers": ToolSpec(
        "top_customers",
        "Top customers by total revenue.",
        {"n": "How many customers to return (default 10)."},
        _t_top_customers,
    ),
    "segment_customers": ToolSpec(
        "segment_customers",
        "RFM segmentation: counts of Champions, Loyal, At-Risk, Lost, etc.",
        {},
        _t_segment_customers,
    ),
    "product_quadrants": ToolSpec(
        "product_quadrants",
        "BCG-style quadrant counts: Star, Cash Cow, Question Mark, Dog.",
        {},
        _t_product_quadrants,
    ),
    "co_purchases": ToolSpec(
        "co_purchases",
        "Most frequent co-purchased product pairs by lift.",
        {"basket_key": "Column that groups items into a basket (e.g. order_id).",
         "n": "How many pairs to return (default 10)."},
        _t_co_purchases,
    ),
    "price_elasticity": ToolSpec(
        "price_elasticity",
        "Per-product price elasticity (log-log regression slope).",
        {},
        _t_price_elasticity,
    ),
    "churn_risk": ToolSpec(
        "churn_risk",
        "Distribution of customer churn risk (Active / Cooling / At-Risk / Churned).",
        {"days": "Churn-window threshold in days (default 60)."},
        _t_churn_risk,
    ),
    "cohort_retention": ToolSpec(
        "cohort_retention",
        "Monthly cohort retention matrix.",
        {},
        _t_cohort_retention,
    ),
    "describe_columns": ToolSpec(
        "describe_columns",
        "Per-column dtype, null percentage, and basic stats. Use when you don't know the schema.",
        {},
        _t_describe_columns,
    ),
}


# --------------------------------------------------------------------------- #
# Plan + result types
# --------------------------------------------------------------------------- #

@dataclass
class PlanStep:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    why: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)
    reasoning: str = ""
    backend: str = "heuristic"   # "llm" or "heuristic"

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps],
                "reasoning": self.reasoning, "backend": self.backend}


@dataclass
class AgentResult:
    question: str
    answer: str
    plan: Plan
    observations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer,
                "plan": self.plan.to_dict(), "observations": self.observations}


# --------------------------------------------------------------------------- #
# Heuristic planner — keyword routing
# --------------------------------------------------------------------------- #

# Each pattern → (tool_name, default_args, friendly_intent). Order matters:
# more-specific first. The `friendly_intent` is the user-facing prose that
# becomes the plan step's `why` — never expose raw regex to the workbench.
_HEURISTIC_PATTERNS: List[tuple[str, str, Dict[str, Any], str]] = [
    # Retention
    (r"\b(churn\w*|leaving|inactive|retention risk)\b",
     "churn_risk", {},
     "score every customer for churn risk based on recency"),
    (r"\b(cohort\w*|monthly retention|retention curve|retention decay|retention by signup)\b",
     "cohort_retention", {},
     "build the signup-cohort retention curve, month by month"),
    # Pricing
    (r"\b(elasticity|elastic|price[\s-]?sensitiv\w*|pricing power|how price[\s-]?sensitive|"
     r"demand\s+drops?\s+when\s+(i\s+)?(raise|increase)\s+(the\s+)?prices?)\b",
     "price_elasticity", {},
     "estimate how demand responds to price changes"),
    # Basket
    (r"\b(co.?purchas\w*|basket\w*|frequently bought|bundle\w*|cross.?sell\w*)\b",
     "co_purchases", {},
     "find the strongest product co-purchase pairs"),
    # BCG quadrants
    (r"\b(quadrants?|stars?|cash\s+cows?|dog\s+products?|bcg|question\s+marks?)\b",
     "product_quadrants", {},
     "place each SKU into a BCG quadrant by share × growth"),
    # RFM/segments
    (r"\b(segment\w*|rfm|champion\w*|loyal(ty)?|at-?risk)\b",
     "segment_customers", {},
     "bucket customers into RFM segments (Champions, Loyal, At-Risk, Lost…)"),
    # Top customers
    (r"\b(?:top|best|biggest|highest)\s+(\d+)?\s*"
     r"(?:customer\w*|spender\w*|paying\s+customer\w*)\b",
     "top_customers", {},
     "rank customers by total revenue"),
    # Top products
    (r"\b(?:best.?selling|top|best|biggest)\s+(\d+)?\s*(?:products?|skus?)\b",
     "top_products", {},
     "rank SKUs by total revenue"),
    (r"\bwhich\s+(\d+)\s+(?:products?|skus?)\b",
     "top_products", {},
     "rank SKUs by total revenue"),
    # Revenue cadence
    (r"\b(daily|by day)\b.*revenue|revenue.*\b(daily|by day)\b",
     "revenue_by_period", {"freq": "D"},
     "aggregate revenue by day"),
    (r"\b(monthly|by month)\b.*\b(revenue|sales)\b|\b(revenue|sales)\b.*\b(monthly|by month)\b",
     "revenue_by_period", {"freq": "M"},
     "aggregate revenue by month"),
    (r"\b(weekly|by week|trend|over time)\b",
     "revenue_by_period", {"freq": "W"},
     "aggregate revenue by week to show the trend"),
    (r"\b(revenue|sales)\b",
     "revenue_by_period", {"freq": "W"},
     "aggregate revenue by week (default cadence)"),
    # Schema
    (r"\b(schema|columns?|what.* in (the|my) (data|dataset)|describe.*columns?)\b",
     "describe_columns", {},
     "list the columns in the dataset with their dtypes"),
]


def _heuristic_plan(question: str) -> Plan:
    """Keyword-route a question to one or more tools.

    Each step's `why` field is composed user-facing prose — it MUST be safe to
    render directly in the workbench. No regex literals, no internal jargon.
    """
    q = question.lower()
    steps: List[PlanStep] = []
    seen = set()
    for pattern, tool, default_args, intent in _HEURISTIC_PATTERNS:
        m = re.search(pattern, q)
        if not (m and tool not in seen):
            continue
        args = dict(default_args)
        # Pull out a number if the pattern captured one (e.g. "top 5 products")
        for grp in m.groups():
            if grp and grp.isdigit():
                args["n"] = int(grp)
                break
        # Build the human "why":
        #   You said "top 5 products" → rank SKUs by total revenue (n=5).
        trigger = m.group(0).strip()
        arg_bits = []
        if "n"    in args: arg_bits.append(f"n={args['n']}")
        if "freq" in args: arg_bits.append({"D": "daily", "W": "weekly", "M": "monthly"}
                                            .get(args["freq"], args["freq"]))
        suffix = f" ({', '.join(arg_bits)})" if arg_bits else ""
        why = f'You said "{trigger}" → {intent}{suffix}.'
        steps.append(PlanStep(tool=tool, args=args, why=why))
        seen.add(tool)

    if not steps:
        steps.append(PlanStep(
            tool="describe_columns", args={},
            why="No keyword in your question matched a tool, so I'm describing "
                "the dataset's columns instead — try rephrasing with words like "
                "'revenue', 'top products', 'churn', 'segments', etc.",
        ))
    return Plan(steps=steps,
                reasoning="Keyword routing — no LLM available.",
                backend="heuristic")


# --------------------------------------------------------------------------- #
# LLM planner (optional — uses agent.explain's Gemini client if configured)
# --------------------------------------------------------------------------- #

def _llm_available() -> bool:
    try:
        from agent.explain import _CLIENT  # type: ignore
        return _CLIENT is not None
    except Exception:
        return False


_LLM_DEBUG = os.getenv("LIVEOPS_LLM_DEBUG", "").strip() in ("1", "true", "yes")


def _llm_warn(msg: str) -> None:
    """One-line stderr log when the LLM planner falls back. Off by default;
    enable with LIVEOPS_LLM_DEBUG=1 for diagnosing 'why is llm == heuristic'."""
    if _LLM_DEBUG:
        import sys
        print(f"[llm-fallback] {msg}", file=sys.stderr)


def _llm_plan(question: str, role_map: Dict[str, str],
              schema_summary: List[Dict[str, Any]]) -> Optional[Plan]:
    """Ask Gemini to pick tools. Returns None on any failure so caller falls
    back to the heuristic planner — never raises.

    Set LIVEOPS_LLM_DEBUG=1 to see one-line stderr diagnostics on each
    fallback (network errors, malformed JSON, etc.)."""
    try:
        from agent.explain import _CLIENT, _get_model_name  # type: ignore
    except Exception as e:
        _llm_warn(f"import agent.explain failed: {type(e).__name__}: {e}")
        return None
    if _CLIENT is None:
        _llm_warn("agent.explain._CLIENT is None (no GEMINI_API_KEY?)")
        return None

    tool_catalog = [t.to_dict() for t in TOOLS.values()]
    system_prompt = (
        "You are an analyst's planning agent. Given a question, the dataset's "
        "role map, and a schema summary, return a JSON plan listing which "
        "tools to call. Available tools are listed below. Only use tool names "
        "from this list. Args must match each tool's params. Return AT MOST "
        "3 tools. Output strictly: "
        '{"steps":[{"tool":"...","args":{...},"why":"..."}],"reasoning":"..."}'
    )
    user_payload = {
        "question": question,
        "role_map": role_map,
        "schema": schema_summary,
        "tools": tool_catalog,
    }

    try:
        resp = _CLIENT.models.generate_content(
            model=_get_model_name(),
            contents=[
                {"role": "user", "parts": [
                    {"text": system_prompt},
                    {"text": json.dumps(user_payload, default=str)},
                ]},
            ],
        )
        text = (resp.text or "").strip()
        # Strip code fences if present
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        data = json.loads(text)
        steps = [PlanStep(tool=s["tool"], args=dict(s.get("args", {})),
                          why=str(s.get("why", "")))
                 for s in data.get("steps", []) if s.get("tool") in TOOLS]
        if not steps:
            _llm_warn(f"LLM returned no usable steps for {question!r}")
            return None
        return Plan(steps=steps[:3],
                    reasoning=str(data.get("reasoning", "")),
                    backend="llm")
    except Exception as e:
        _llm_warn(f"LLM call failed for {question!r}: {type(e).__name__}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Synthesizer — turns observations into a natural answer
# --------------------------------------------------------------------------- #

def _format_observation(obs: Dict[str, Any], max_lines: int = 6) -> str:
    """Render an observation compactly for the answer text."""
    if "error" in obs.get("result", {}):
        return f"  · {obs['tool']}: {obs['result']['error']}"
    res = obs["result"]
    if isinstance(res, dict):
        items = list(res.items())[:max_lines]
        body = ", ".join(f"{k}: {v}" for k, v in items)
        return f"  · {obs['tool']}: {body}"
    if isinstance(res, list):
        if not res:
            return f"  · {obs['tool']}: (no rows)"
        sample = res[:max_lines]
        if isinstance(sample[0], dict):
            keys = list(sample[0].keys())[:3]
            rows = "; ".join(
                ", ".join(f"{k}={r.get(k)}" for k in keys) for r in sample
            )
            return f"  · {obs['tool']} ({len(res)} rows): {rows}"
        return f"  · {obs['tool']}: {sample}"
    return f"  · {obs['tool']}: {res}"


def _heuristic_synthesize(question: str, observations: List[Dict[str, Any]]) -> str:
    if not observations:
        return ("I couldn't find a relevant tool to answer that. "
                "Try asking about revenue, top products, churn, segments, or elasticity.")
    head = f"Here's what I found for: *{question.strip()}*"
    body = "\n".join(_format_observation(o) for o in observations)
    return head + "\n\n" + body


def _llm_synthesize(question: str, observations: List[Dict[str, Any]]) -> Optional[str]:
    try:
        from agent.explain import _CLIENT, _get_model_name  # type: ignore
    except Exception:
        return None
    if _CLIENT is None:
        return None
    try:
        prompt = (
            "You are a data analyst. Given a question and the JSON results of "
            "tools that were run, write a concise answer in 2-4 sentences. "
            "Cite the specific numbers. Don't apologize, don't repeat the question."
        )
        payload = {"question": question, "observations": observations}
        resp = _CLIENT.models.generate_content(
            model=_get_model_name(),
            contents=[{"role": "user", "parts": [
                {"text": prompt},
                {"text": json.dumps(payload, default=str)},
            ]}],
        )
        return (resp.text or "").strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Executor + public ask()
# --------------------------------------------------------------------------- #

def _schema_summary(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for c in df.columns[:30]:  # cap so prompt stays small
        out.append({"col": c, "dtype": str(df[c].dtype),
                    "sample": str(df[c].dropna().head(2).tolist())})
    return out


def _execute(plan: Plan, df: pd.DataFrame,
             role_map: Dict[str, str]) -> List[Dict[str, Any]]:
    obs: List[Dict[str, Any]] = []
    for step in plan.steps:
        spec = TOOLS.get(step.tool)
        if spec is None:
            obs.append({"tool": step.tool, "args": step.args,
                        "result": {"error": f"unknown tool {step.tool}"}})
            continue
        try:
            result = spec.fn(df, role_map, **step.args)
        except TypeError as e:
            # Filter out unsupported kwargs and retry once — protects against
            # an LLM hallucinating a param the underlying tool doesn't accept.
            try:
                import inspect
                sig = inspect.signature(spec.fn)
                valid = {k: v for k, v in step.args.items()
                         if k in sig.parameters}
                result = spec.fn(df, role_map, **valid)
            except Exception as inner:
                result = {"error": f"{type(inner).__name__}: {inner}"}
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
        obs.append({"tool": step.tool, "args": step.args, "result": result})
    return obs


def ask(question: str, df: pd.DataFrame,
        role_map: Optional[Dict[str, str]] = None,
        *, backend: str = "auto") -> AgentResult:
    """Plan → execute → synthesize.

    `backend`:
      - "auto"     — use LLM if available, else heuristic
      - "llm"      — force LLM (returns heuristic plan if not configured)
      - "heuristic"— never call the LLM

    Always returns an AgentResult; never raises.
    """
    role_map = role_map or {}
    plan: Optional[Plan] = None

    if backend in ("auto", "llm") and _llm_available():
        plan = _llm_plan(question, role_map, _schema_summary(df))
    if plan is None:
        plan = _heuristic_plan(question)

    obs = _execute(plan, df, role_map)

    answer: Optional[str] = None
    if backend in ("auto", "llm") and plan.backend == "llm":
        answer = _llm_synthesize(question, obs)
    if not answer:
        answer = _heuristic_synthesize(question, obs)

    return AgentResult(question=question, answer=answer, plan=plan, observations=obs)


__all__ = ["TOOLS", "ToolSpec", "Plan", "PlanStep", "AgentResult", "ask"]
