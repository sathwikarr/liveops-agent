"""Pinboard — a saveable collection of charts assembled from the analyst page.

Storage shape: a list of `PinSpec` dicts, each one a builder name plus the
JSON-safe params needed to re-render. We deliberately do NOT pickle Plotly
Figure objects — keeping specs as data means a session reload still produces
live, interactive charts and we can roundtrip the whole pinboard through JSON.

Public API:
- PinSpec: dataclass with kind, title, params, created_at
- add_pin(pinboard, spec) -> list (returns new pinboard, no mutation)
- remove_pin(pinboard, idx) -> list
- render_pin(spec, ctx) -> plotly Figure (looks up the builder in
  analyst.charts.REGISTRY, calls it with params expanded from `ctx`)
- export_html(pinboard, ctx, title="Analyst pinboard") -> str (standalone HTML)
- to_json(pinboard) / from_json(s) — persistence helpers

`ctx` is the dict of live DataFrames the page already has (rfm_df, matrix_df,
etc.). Pin params reference dataframes by name so the same pin renders
correctly after a re-ingest.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import plotly.io as pio

from analyst.charts import REGISTRY


@dataclass
class PinSpec:
    kind: str                  # builder key in charts.REGISTRY
    title: str                 # user-visible label
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat(timespec="seconds")
        if self.kind not in REGISTRY:
            raise ValueError(f"Unknown chart kind: {self.kind}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PinSpec":
        return cls(kind=d["kind"], title=d["title"],
                   params=dict(d.get("params", {})),
                   created_at=d.get("created_at", ""))


# --------------------------------------------------------------------------- #
# CRUD (pure functions — easy to test, no global state)
# --------------------------------------------------------------------------- #

def add_pin(pinboard: List[PinSpec], spec: PinSpec) -> List[PinSpec]:
    """Append spec, deduping by (kind, title) — same chart pinned twice
    just refreshes the entry rather than stacking duplicates."""
    out = [p for p in pinboard if not (p.kind == spec.kind and p.title == spec.title)]
    out.append(spec)
    return out


def remove_pin(pinboard: List[PinSpec], idx: int) -> List[PinSpec]:
    if idx < 0 or idx >= len(pinboard):
        return list(pinboard)
    return pinboard[:idx] + pinboard[idx + 1:]


def move_pin(pinboard: List[PinSpec], idx: int, delta: int) -> List[PinSpec]:
    """Reorder a pin (delta=-1 to move up, +1 to move down)."""
    new_idx = max(0, min(len(pinboard) - 1, idx + delta))
    if new_idx == idx:
        return list(pinboard)
    out = list(pinboard)
    out.insert(new_idx, out.pop(idx))
    return out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _resolve_params(params: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Expand `{"$ref": "rfm_df"}` references against the live ctx dict.

    Plain values pass through. This lets pin specs stay JSON-friendly
    while still re-binding to fresh DataFrames after a re-ingest.
    """
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, dict) and "$ref" in v:
            out[k] = ctx.get(v["$ref"])
        else:
            out[k] = v
    return out


def render_pin(spec: PinSpec, ctx: Dict[str, Any]) -> go.Figure:
    builder = REGISTRY[spec.kind]
    resolved = _resolve_params(spec.params, ctx)
    return builder(**resolved)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def export_html(pinboard: List[PinSpec], ctx: Dict[str, Any],
                title: str = "Analyst pinboard") -> str:
    """Render every pin to a single standalone HTML page.

    Plotly's CDN script is inlined once; subsequent figures embed only their
    JSON spec. The result is a single file that opens anywhere — perfect for
    sharing with a recruiter or a stakeholder.
    """
    parts: List[str] = [
        "<!doctype html>",
        f"<html><head><meta charset='utf-8'><title>{title}</title>",
        "<style>",
        "body{font-family:Inter,system-ui,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:24px;}",
        ".pin{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px;margin-bottom:18px;box-shadow:0 1px 2px rgba(0,0,0,0.04);}",
        ".pin h2{font-size:15px;margin:0 0 6px;color:#1e293b;}",
        ".meta{font-size:12px;color:#64748b;margin-bottom:10px;}",
        "h1{font-size:22px;margin:0 0 18px;}",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<div class='meta'>Generated {datetime.utcnow().isoformat(timespec='seconds')} UTC · {len(pinboard)} chart(s)</div>",
    ]

    if not pinboard:
        parts.append("<p>No pinned charts yet. Pin charts from the analyst page to build a dashboard.</p>")
    else:
        # First pin includes the plotly.js CDN, subsequent ones don't.
        for i, spec in enumerate(pinboard):
            try:
                fig = render_pin(spec, ctx)
            except Exception as e:
                parts.append(
                    f"<div class='pin'><h2>{spec.title}</h2>"
                    f"<div class='meta'>Failed to render ({type(e).__name__}: {e})</div></div>"
                )
                continue
            include_js = "cdn" if i == 0 else False
            chart_html = pio.to_html(fig, include_plotlyjs=include_js,
                                     full_html=False, config={"displayModeBar": False})
            parts.append(
                f"<div class='pin'><h2>{spec.title}</h2>"
                f"<div class='meta'>{spec.kind} · pinned {spec.created_at}</div>"
                f"{chart_html}</div>"
            )
    parts.append("</body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def to_json(pinboard: List[PinSpec]) -> str:
    return json.dumps([p.to_dict() for p in pinboard], indent=2, default=str)


def from_json(s: str) -> List[PinSpec]:
    if not s or not s.strip():
        return []
    raw = json.loads(s)
    return [PinSpec.from_dict(d) for d in raw]


__all__ = ["PinSpec", "add_pin", "remove_pin", "move_pin",
           "render_pin", "export_html", "to_json", "from_json"]
