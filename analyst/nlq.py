"""Natural-language query — turn English into a pandas operation.

Two execution modes:
- LLM mode (preferred): Gemini returns a pandas expression bounded to a
  whitelist of safe ops. We `eval` it inside a sandboxed namespace.
- Heuristic fallback: pattern-matched answers for common questions
  (top product, total revenue, monthly trend, ...) so the feature still
  works without an API key.

Returns a `NLQResult` containing both an `answer` string and an optional
`data` DataFrame the UI can render as a table/chart.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class NLQResult:
    answer: str
    expression: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    method: str = "heuristic"

    def to_dict(self) -> dict:
        return {"answer": self.answer, "expression": self.expression,
                "method": self.method,
                "data_preview": self.data.head(20).to_dict("records") if self.data is not None else None}


# Safe pandas-only namespace for eval().
_SAFE_BUILTINS = {"len": len, "sum": sum, "min": min, "max": max,
                  "abs": abs, "round": round, "sorted": sorted}


def _safe_eval(expr: str, df: pd.DataFrame) -> object:
    if any(bad in expr for bad in ("__", "import", "open(", "eval(", "exec(",
                                   "compile(", "system", "subprocess")):
        raise ValueError("Refusing to evaluate unsafe expression.")
    return eval(expr, {"__builtins__": _SAFE_BUILTINS}, {"df": df, "pd": pd})


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #

_TOP_RE = re.compile(r"top\s+(\d+)?\s*(product|customer|region|category)?", re.I)
_TOTAL_RE = re.compile(r"total\s+(revenue|sales|orders|amount)", re.I)
_MONTHLY_RE = re.compile(r"\b(by month|per month|monthly|month over month|mom)\b", re.I)


def _heuristic_answer(query: str, df: pd.DataFrame, role_map: dict) -> Optional[NLQResult]:
    q = query.lower()

    if m := _TOTAL_RE.search(q):
        amt = role_map.get("amount")
        if amt and amt in df.columns:
            total = pd.to_numeric(df[amt], errors="coerce").sum()
            return NLQResult(
                answer=f"Total {m.group(1)}: {total:,.2f}",
                expression=f"df['{amt}'].sum()",
                data=pd.DataFrame({"metric": [f"total_{m.group(1)}"], "value": [round(float(total), 2)]}),
            )

    if m := _TOP_RE.search(q):
        n = int(m.group(1)) if m.group(1) else 5
        what = (m.group(2) or "").lower() or "product"
        role = {"product": "product", "customer": "customer", "region": "region"}.get(what)
        if role and role in role_map and role_map[role] in df.columns:
            col = role_map[role]
            amt = role_map.get("amount")
            if amt and amt in df.columns:
                grp = df.assign(_amt=pd.to_numeric(df[amt], errors="coerce")) \
                       .groupby(col)["_amt"].sum().sort_values(ascending=False).head(n)
                tbl = grp.reset_index().rename(columns={"_amt": amt})
                top = grp.index[0] if not grp.empty else "—"
                return NLQResult(
                    answer=f"Top {n} {what} by {amt}: {top} leads with {grp.iloc[0]:,.2f}.",
                    expression=f"df.groupby('{col}')['{amt}'].sum().sort_values(ascending=False).head({n})",
                    data=tbl,
                )

    if _MONTHLY_RE.search(q):
        amt = role_map.get("amount")
        date = role_map.get("date")
        if amt and date and {amt, date}.issubset(df.columns):
            sub = df[[date, amt]].dropna().copy()
            sub[date] = pd.to_datetime(sub[date], errors="coerce")
            sub[amt] = pd.to_numeric(sub[amt], errors="coerce")
            sub = sub.dropna()
            if sub.empty:
                return NLQResult(answer="No valid date+amount rows to aggregate.", method="heuristic")
            monthly = sub.set_index(date)[amt].resample("MS").sum().rename(amt).reset_index()
            return NLQResult(
                answer=f"Monthly {amt}: {len(monthly)} months covered.",
                expression=f"df.set_index('{date}')['{amt}'].resample('MS').sum()",
                data=monthly,
            )
    return None


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #

_LLM_SYSTEM = """You are a pandas expert. Translate the user's question into ONE pandas
expression that operates on a DataFrame named `df`. Return ONLY the expression
on a single line — no markdown, no commentary, no semicolons. The expression
must be safe (no imports, no I/O, no eval). Use only df, pd, and standard
pandas methods. Cast text columns with pd.to_numeric / pd.to_datetime where
needed."""


def _llm_expression(query: str, df: pd.DataFrame) -> Optional[str]:
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            return None
        from agent.explain import _client  # type: ignore
        from google.genai import types  # type: ignore
        client = _client()
        if client is None:
            return None
        cols = list(df.columns)[:50]
        prompt = (
            f"{_LLM_SYSTEM}\n\nColumns available: {cols}\n"
            f"Sample row: {df.head(1).to_dict('records')}\n"
            f"Question: {query}\nExpression:"
        )
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        text = (resp.text or "").strip().strip("`")
        # Strip language hint from a fenced block if it slipped through.
        if text.startswith("python"):
            text = text[len("python"):].strip()
        return text.splitlines()[0].strip() if text else None
    except Exception:
        return None


def ask(query: str, df: pd.DataFrame, role_map: Optional[dict] = None) -> NLQResult:
    role_map = role_map or {}
    # Heuristic first — instant answers for the common questions.
    h = _heuristic_answer(query, df, role_map)
    if h is not None:
        return h

    expr = _llm_expression(query, df)
    if not expr:
        return NLQResult(
            answer="I couldn't translate that into a query. Try asking about totals, top-N, or monthly trends.",
            method="heuristic",
        )

    try:
        result = _safe_eval(expr, df)
    except Exception as e:
        return NLQResult(
            answer=f"Generated expression failed: {type(e).__name__}: {e}",
            expression=expr, method="llm",
        )

    if isinstance(result, pd.DataFrame):
        return NLQResult(answer=f"Returned a {len(result)}-row table.",
                         expression=expr, data=result, method="llm")
    if isinstance(result, pd.Series):
        out_df = result.reset_index().rename(columns={result.name or 0: "value"})
        return NLQResult(answer=f"Returned a {len(result)}-value series.",
                         expression=expr, data=out_df, method="llm")
    return NLQResult(answer=str(result), expression=expr, method="llm")


__all__ = ["ask", "NLQResult"]
