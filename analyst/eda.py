"""Auto-EDA — distributions, correlations, outliers, seasonality, narrative.

Public API:
- `profile(df, schema)` -> EDAReport
    A pure-Python report (no matplotlib calls). The Streamlit page consumes
    `report.dict()` and renders charts itself.

- `narrate(report, kind)` -> str
    Plain-English summary written by Gemini (or a deterministic stub when no
    GEMINI_API_KEY is set). Reuses agent/explain.py's client.

The split between numeric data + rendering keeps this layer testable: pytest
exercises `profile()` end-to-end without a display.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from analyst.ingest import Schema

# --------------------------------------------------------------------------- #
# Per-column distribution stats
# --------------------------------------------------------------------------- #

@dataclass
class NumericDist:
    name: str
    count: int
    mean: float
    std: float
    min_: float
    p25: float
    median: float
    p75: float
    max_: float
    skew: float
    null_pct: float
    histogram: list[int]
    bin_edges: list[float]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["min"] = d.pop("min_"); d["max"] = d.pop("max_")
        return d


@dataclass
class CategoricalDist:
    name: str
    count: int
    n_unique: int
    null_pct: float
    top: list[tuple[str, int]]  # value, count

    def to_dict(self) -> dict:
        return {"name": self.name, "count": self.count, "n_unique": self.n_unique,
                "null_pct": self.null_pct, "top": self.top}


@dataclass
class DateRange:
    name: str
    min_ts: Optional[str]
    max_ts: Optional[str]
    span_days: float
    null_pct: float

    def to_dict(self) -> dict:
        return self.__dict__


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #

@dataclass
class OutlierRow:
    column: str
    method: str
    n_outliers: int
    pct: float
    examples: list[float]

    def to_dict(self) -> dict:
        return self.__dict__


def _iqr_outliers(s: pd.Series) -> tuple[int, list[float]]:
    s = s.dropna()
    if len(s) < 8:
        return 0, []
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, []
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    out = s[mask]
    return int(mask.sum()), out.head(5).round(3).tolist()


def _z_outliers(s: pd.Series, threshold: float = 3.0) -> tuple[int, list[float]]:
    s = s.dropna()
    if len(s) < 8 or s.std() == 0:
        return 0, []
    z = (s - s.mean()) / s.std()
    mask = z.abs() > threshold
    return int(mask.sum()), s[mask].head(5).round(3).tolist()


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #

@dataclass
class Seasonality:
    has_weekly: bool
    weekly_strength: float    # max(weekday) / min(weekday) ratio of mean amount
    has_monthly: bool
    monthly_strength: float
    peak_weekday: Optional[str]
    peak_month: Optional[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return self.__dict__


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _seasonality(df: pd.DataFrame, date_col: Optional[str], amount_col: Optional[str]) -> Seasonality:
    notes: list[str] = []
    if not date_col or not amount_col or date_col not in df.columns or amount_col not in df.columns:
        return Seasonality(False, 0.0, False, 0.0, None, None,
                           ["Need both a date and an amount column to detect seasonality."])
    s = df[[date_col, amount_col]].dropna()
    s = s[pd.to_numeric(s[amount_col], errors="coerce").notna()]
    if len(s) < 14:
        return Seasonality(False, 0.0, False, 0.0, None, None,
                           ["Not enough rows (<14) to detect seasonality."])
    s[amount_col] = pd.to_numeric(s[amount_col])
    s["_dow"] = s[date_col].dt.dayofweek
    s["_mo"] = s[date_col].dt.month

    by_dow = s.groupby("_dow")[amount_col].mean()
    by_mo = s.groupby("_mo")[amount_col].mean()

    def _strength(series: pd.Series) -> float:
        lo, hi = series.min(), series.max()
        return float(hi / lo) if lo and lo > 0 else float("inf") if hi else 0.0

    w_strength = _strength(by_dow) if not by_dow.empty else 0.0
    m_strength = _strength(by_mo) if not by_mo.empty else 0.0

    has_weekly = math.isfinite(w_strength) and w_strength >= 1.4
    has_monthly = math.isfinite(m_strength) and m_strength >= 1.4

    peak_dow = _WEEKDAYS[int(by_dow.idxmax())] if has_weekly and not by_dow.empty else None
    peak_mo = _MONTHS[int(by_mo.idxmax()) - 1] if has_monthly and not by_mo.empty else None

    if has_weekly:
        notes.append(f"Weekly peak on {peak_dow} (×{w_strength:.2f} the slowest day).")
    if has_monthly:
        notes.append(f"Monthly peak in {peak_mo} (×{m_strength:.2f} the slowest month).")

    return Seasonality(has_weekly, float(w_strength) if math.isfinite(w_strength) else 0.0,
                       has_monthly, float(m_strength) if math.isfinite(m_strength) else 0.0,
                       peak_dow, peak_mo, notes)


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #

@dataclass
class CorrelationPair:
    a: str
    b: str
    pearson: float

    def to_dict(self) -> dict:
        return self.__dict__


def _correlations(df: pd.DataFrame, numeric_cols: list[str], top_n: int = 10) -> list[CorrelationPair]:
    if len(numeric_cols) < 2:
        return []
    sub = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="pearson", min_periods=8)
    pairs: list[CorrelationPair] = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append(CorrelationPair(a, b, float(v)))
    pairs.sort(key=lambda p: abs(p.pearson), reverse=True)
    return pairs[:top_n]


# --------------------------------------------------------------------------- #
# Top-level report
# --------------------------------------------------------------------------- #

@dataclass
class EDAReport:
    n_rows: int
    n_cols: int
    numeric: list[NumericDist] = field(default_factory=list)
    categorical: list[CategoricalDist] = field(default_factory=list)
    dates: list[DateRange] = field(default_factory=list)
    correlations: list[CorrelationPair] = field(default_factory=list)
    outliers: list[OutlierRow] = field(default_factory=list)
    seasonality: Optional[Seasonality] = None
    headline: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "n_cols": self.n_cols,
            "numeric": [d.to_dict() for d in self.numeric],
            "categorical": [d.to_dict() for d in self.categorical],
            "dates": [d.to_dict() for d in self.dates],
            "correlations": [c.to_dict() for c in self.correlations],
            "outliers": [o.to_dict() for o in self.outliers],
            "seasonality": self.seasonality.to_dict() if self.seasonality else None,
            "headline": self.headline,
        }


def _numeric_dist(df: pd.DataFrame, col: str) -> Optional[NumericDist]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    null_pct = 1 - len(s) / max(len(df), 1)
    hist, edges = np.histogram(s, bins=min(20, max(5, int(np.sqrt(len(s))))))
    return NumericDist(
        name=col, count=int(len(s)),
        mean=float(s.mean()), std=float(s.std(ddof=0)),
        min_=float(s.min()), p25=float(s.quantile(0.25)),
        median=float(s.median()), p75=float(s.quantile(0.75)),
        max_=float(s.max()),
        skew=float(s.skew()) if len(s) > 2 else 0.0,
        null_pct=null_pct,
        histogram=hist.tolist(),
        bin_edges=[float(x) for x in edges],
    )


def _categorical_dist(df: pd.DataFrame, col: str) -> Optional[CategoricalDist]:
    s = df[col].dropna().astype(str)
    if s.empty:
        return None
    counts = s.value_counts().head(10)
    return CategoricalDist(
        name=col, count=int(len(s)),
        n_unique=int(s.nunique()),
        null_pct=1 - len(s) / max(len(df), 1),
        top=[(str(k), int(v)) for k, v in counts.items()],
    )


def _date_range(df: pd.DataFrame, col: str) -> Optional[DateRange]:
    s = df[col].dropna()
    if s.empty:
        return None
    span = (s.max() - s.min()).total_seconds() / 86400.0
    return DateRange(
        name=col,
        min_ts=str(s.min()), max_ts=str(s.max()),
        span_days=float(span),
        null_pct=1 - len(s) / max(len(df), 1),
    )


def _build_headline(report: EDAReport, kind: Optional[str], role_map: Optional[dict] = None) -> list[str]:
    role_map = role_map or {}
    lines: list[str] = []
    lines.append(f"Dataset has {report.n_rows:,} rows × {report.n_cols} columns.")
    if report.dates:
        d = report.dates[0]
        lines.append(f"Time range: {d.min_ts} → {d.max_ts} ({d.span_days:.1f} days).")
    if kind:
        lines.append(f"Looks like a {kind} dataset.")
    if report.seasonality and report.seasonality.notes:
        lines.append(report.seasonality.notes[0])
    if report.correlations:
        c = report.correlations[0]
        lines.append(f"Strongest correlation: {c.a} vs {c.b} (r = {c.pearson:+.2f}).")
    big_outliers = [o for o in report.outliers if o.pct > 0.02]
    if big_outliers:
        o = big_outliers[0]
        lines.append(f"{o.column!r} has {o.n_outliers} outliers ({o.pct:.1%} of rows) by {o.method}.")
    return lines


def profile(df: pd.DataFrame, schema: Schema, kind: Optional[str] = None,
            role_map: Optional[dict] = None) -> EDAReport:
    """Compute distributions, correlations, outliers, and seasonality."""
    report = EDAReport(n_rows=len(df), n_cols=df.shape[1])

    for col in schema.numeric_cols():
        d = _numeric_dist(df, col)
        if d: report.numeric.append(d)

    for col in schema.categorical_cols():
        d = _categorical_dist(df, col)
        if d: report.categorical.append(d)

    for col in schema.datetime_cols():
        d = _date_range(df, col)
        if d: report.dates.append(d)

    report.correlations = _correlations(df, schema.numeric_cols(), top_n=10)

    for col in schema.numeric_cols():
        s = pd.to_numeric(df[col], errors="coerce")
        n_iqr, ex_iqr = _iqr_outliers(s)
        if n_iqr:
            report.outliers.append(OutlierRow(col, "IQR", n_iqr,
                                              n_iqr / max(len(df), 1), ex_iqr))
        n_z, ex_z = _z_outliers(s)
        if n_z and n_z != n_iqr:
            report.outliers.append(OutlierRow(col, "Z>3", n_z,
                                              n_z / max(len(df), 1), ex_z))

    role_map = role_map or {}
    report.seasonality = _seasonality(df, role_map.get("date"), role_map.get("amount"))
    report.headline = _build_headline(report, kind, role_map)
    return report


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #

def _stub_narrative(report: EDAReport, kind: Optional[str]) -> str:
    parts = list(report.headline)
    if report.numeric:
        n = report.numeric[0]
        parts.append(f"{n.name!r} ranges from {n.min_:,.2f} to {n.max_:,.2f} "
                     f"with a median of {n.median:,.2f}.")
    if report.categorical:
        c = report.categorical[0]
        if c.top:
            parts.append(f"{c.name!r} is dominated by {c.top[0][0]!r} ({c.top[0][1]:,} rows).")
    return " ".join(parts)


def narrate(report: EDAReport, kind: Optional[str] = None) -> str:
    """Plain-English EDA summary. Uses Gemini if configured; otherwise stub."""
    try:
        import os
        if not os.environ.get("GEMINI_API_KEY"):
            return _stub_narrative(report, kind)
        from agent.explain import _client  # type: ignore
        client = _client()
        if client is None:
            return _stub_narrative(report, kind)
        from google.genai import types  # type: ignore
        prompt = (
            "You are a senior analyst. Summarize this dataset in 4-6 short sentences "
            "for a business owner. Lead with the biggest finding. No bullets.\n\n"
            f"DATASET KIND: {kind or 'unknown'}\n"
            f"REPORT: {report.to_dict()}"
        )
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        text = (resp.text or "").strip()
        return text or _stub_narrative(report, kind)
    except Exception:
        return _stub_narrative(report, kind)


__all__ = ["profile", "narrate", "EDAReport",
           "NumericDist", "CategoricalDist", "DateRange",
           "OutlierRow", "Seasonality", "CorrelationPair"]
