"""LLM-powered anomaly explanations.

Uses the new `google-genai` SDK (the old `google-generativeai` package is
deprecated and prints FutureWarnings). Returns either:

- `explain_anomaly(...)`: a 3-bullet markdown string (back-compat for the
  dashboard / runner that already render this).
- `explain_anomaly_structured(...)`: a dict with `cause`, `severity`,
  `recommended_action`, `bullets` — used by the action layer to weight
  notifications and store machine-readable rationale.

Falls back to deterministic stub output when no API key is configured so the
rest of the pipeline keeps working in tests.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

from dotenv import find_dotenv, load_dotenv

from agent.utils import with_backoff

# Optional Streamlit (for secrets + quiet warnings)
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    HAS_STREAMLIT = False

# New SDK
try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
    HAS_GENAI = True
except Exception:
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    HAS_GENAI = False


SEVERITY_VALUES = ("low", "medium", "high", "critical")
DEFAULT_MODEL = "gemini-2.5-flash"

# JSON schema for structured output. Keep it small so Flash stays cheap.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "cause": {"type": "string", "description": "1-sentence root cause hypothesis."},
        "severity": {"type": "string", "enum": list(SEVERITY_VALUES)},
        "recommended_action": {
            "type": "string",
            "description": "1-line concrete action a human ops engineer should take.",
        },
        "bullets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exactly 3 short bullets the dashboard can render.",
        },
    },
    "required": ["cause", "severity", "recommended_action", "bullets"],
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def _get_api_key() -> Optional[str]:
    load_dotenv(find_dotenv(), override=False)
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    if HAS_STREAMLIT:
        try:
            return st.secrets["GEMINI_API_KEY"]  # type: ignore[attr-defined]
        except Exception:
            pass
    return None


def _get_model_name() -> str:
    if HAS_STREAMLIT:
        try:
            m = st.secrets["GEMINI_MODEL"]  # type: ignore[attr-defined]
            if m:
                return str(m).strip()
        except Exception:
            pass
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()


_API_KEY = _get_api_key()
_CLIENT = None
if HAS_GENAI and _API_KEY:
    try:
        _CLIENT = genai.Client(api_key=_API_KEY)
    except Exception as e:
        if HAS_STREAMLIT:
            try:
                st.warning(f"⚠️ Could not init Gemini client: {e}")
            except Exception:
                pass
        _CLIENT = None


# --------------------------------------------------------------------------- #
# Fallbacks
# --------------------------------------------------------------------------- #

def _fallback_struct(region: str, product_id: str, revenue: float) -> dict:
    """Deterministic stub when no API key / SDK / model available."""
    return {
        "cause": f"Unusual revenue ({revenue:.2f}) for {product_id} in {region}.",
        "severity": "medium",
        "recommended_action": "Review pricing/inventory and check for promo or feed errors.",
        "bullets": [
            f"Sudden regional demand spike in {region}.",
            f"Pricing or promo change for {product_id}.",
            f"Bulk/enterprise purchase or reporting anomaly near {revenue:.2f}.",
        ],
    }


def _bullets_to_markdown(bullets: list[str]) -> str:
    cleaned = []
    for b in bullets[:3]:
        b = (b or "").strip()
        cleaned.append(b if b.startswith(("-", "*", "•")) else f"- {b}")
    while len(cleaned) < 3:
        cleaned.append("- (no additional cause identified)")
    return "\n".join(cleaned[:3])


# --------------------------------------------------------------------------- #
# Core call (cached)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=512)
def _explain_struct_cached(
    region: str,
    product_id: str,
    orders: int,
    inventory: int,
    revenue: float,
    model_name: str,
) -> str:
    """Returns a JSON string. Cached so Streamlit reruns don't re-spend tokens."""
    if not (HAS_GENAI and _CLIENT):
        return json.dumps(_fallback_struct(region, product_id, revenue))

    prompt = f"""You are an AI operations analyst.

An anomaly was detected:
- Region: {region}
- Product ID: {product_id}
- Orders: {orders}
- Inventory: {inventory}
- Revenue: {revenue:.2f}

Return JSON matching the schema with a 1-sentence root cause, a severity
(low/medium/high/critical), a 1-line recommended human action, and exactly
3 short cause bullets."""

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
        temperature=0.3,
    )

    def _call():
        return _CLIENT.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

    try:
        resp = with_backoff(_call)
        if resp is None:
            return json.dumps(_fallback_struct(region, product_id, revenue))
        text = getattr(resp, "text", "") or ""
        # Validate it parses; otherwise fall back
        json.loads(text)
        return text
    except Exception as e:
        if HAS_STREAMLIT:
            try:
                st.warning(f"⚠️ Gemini error: {str(e)[:180]}… — using fallback.")
            except Exception:
                pass
        return json.dumps(_fallback_struct(region, product_id, revenue))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _normalize_inputs(region, product_id, orders, inventory, revenue):
    if not (region and product_id is not None and orders is not None
            and inventory is not None and revenue is not None):
        return None
    try:
        orders = int(orders)
    except Exception:
        orders = 0
    try:
        inventory = int(inventory)
    except Exception:
        inventory = 0
    try:
        revenue = float(revenue)
    except Exception:
        revenue = 0.0
    return region, product_id, orders, inventory, revenue


def explain_anomaly_structured(region, product_id, orders, inventory, revenue) -> dict:
    """Return the parsed structured explanation dict."""
    norm = _normalize_inputs(region, product_id, orders, inventory, revenue)
    if norm is None:
        return _fallback_struct(str(region), str(product_id), 0.0)
    region, product_id, orders, inventory, revenue = norm
    raw = _explain_struct_cached(region, product_id, orders, inventory, revenue, _get_model_name())
    try:
        data = json.loads(raw)
    except Exception:
        data = _fallback_struct(region, product_id, revenue)
    # Normalize severity
    if data.get("severity") not in SEVERITY_VALUES:
        data["severity"] = "medium"
    if not isinstance(data.get("bullets"), list):
        data["bullets"] = _fallback_struct(region, product_id, revenue)["bullets"]
    return data


def explain_anomaly(region, product_id, orders, inventory, revenue) -> str:
    """Back-compat: return the 3-bullet markdown string."""
    if not (region and product_id is not None and orders is not None
            and inventory is not None and revenue is not None):
        return "⚠️ Error: Missing required data for anomaly explanation."
    data = explain_anomaly_structured(region, product_id, orders, inventory, revenue)
    return _bullets_to_markdown(data.get("bullets", []))


if __name__ == "__main__":
    print(json.dumps(
        explain_anomaly_structured("East", "P127", 11, 34, 5406.27),
        indent=2,
    ))
