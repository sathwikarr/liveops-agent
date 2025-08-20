# agent/explain.py
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from agent.utils import with_backoff  # backoff for 429s

# Streamlit is optional (for secrets + quiet UI warnings)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    st = None
    HAS_STREAMLIT = False

import google.generativeai as genai


def _get_gemini_api_key() -> Optional[str]:
    """Prefer .env (works locally), then Streamlit secrets (deploy)."""
    load_dotenv(find_dotenv(), override=False)
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    if HAS_STREAMLIT:
        try:
            return st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    return None


def _get_model_name() -> str:
    """Optional override via env/secrets; defaults to gemini-1.5-flash."""
    if HAS_STREAMLIT:
        try:
            model = st.secrets["GEMINI_MODEL"]
            if model:
                return str(model).strip()
        except Exception:
            pass
    return os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()


# Configure Gemini once (if key exists)
_API_KEY = _get_gemini_api_key()
if _API_KEY:
    genai.configure(api_key=_API_KEY)


def _normalize_to_three_bullets(text: str,
                                region: str,
                                product_id: str,
                                revenue: float) -> str:
    """Ensure we always return exactly 3 bullet lines."""
    text = (text or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        bullets.append(ln if ln.startswith(("-", "*", "•")) else f"- {ln}")
        if len(bullets) == 3:
            break

    while len(bullets) < 3:
        if len(bullets) == 0:
            bullets.append(f"- Sudden regional demand spike in {region}.")
        elif len(bullets) == 1:
            bullets.append(f"- Pricing or promo change for {product_id}.")
        else:
            bullets.append(f"- Bulk/enterprise purchase or reporting anomaly near {revenue:.2f}.")

    return "\n".join(bullets[:3])


@lru_cache(maxsize=512)
def _explain_once(region: str,
                  product_id: str,
                  orders: int,
                  inventory: int,
                  revenue: float,
                  model_name: str) -> str:
    """
    Single call to Gemini with LRU cache (stable across Streamlit reruns).
    Cache key = full argument tuple.
    """
    # If no key configured, skip API and return stable mock
    if not _API_KEY:
        return _normalize_to_three_bullets("", region, product_id, revenue)

    prompt = f"""
You are an AI operations analyst.

An anomaly was detected:
- Region: {region}
- Product ID: {product_id}
- Orders: {orders}
- Inventory: {inventory}
- Revenue: {revenue:.2f}

Return exactly 3 concise, business-focused possible causes as bullet points.
Do not add any preface or suffix text—just the bullet list.
"""

    try:
        model = genai.GenerativeModel(model_name)
        # Wrap with backoff to handle 429/rate limits
        resp = with_backoff(lambda: model.generate_content(prompt))

        if resp is None:
            # backoff exhausted → deterministic fallback
            return _normalize_to_three_bullets("", region, product_id, revenue)

        return _normalize_to_three_bullets(getattr(resp, "text", "") or "", region, product_id, revenue)

    except Exception as e:
        if HAS_STREAMLIT:
            try:
                st.warning(f"⚠️ Gemini error: {str(e)[:180]}... — using fallback.")
            except Exception:
                pass
        return _normalize_to_three_bullets("", region, product_id, revenue)


def explain_anomaly(region, product_id, orders, inventory, revenue) -> str:
    """
    Public API used by the dashboard/runner.
    Returns exactly 3 bullet lines.
    """
    if not (region and product_id and orders is not None and inventory is not None and revenue is not None):
        return "⚠️ Error: Missing required data for anomaly explanation."

    # Normalize types so cache keys are stable
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

    model_name = _get_model_name()
    return _explain_once(region, product_id, orders, inventory, revenue, model_name)


# Quick local test
if __name__ == "__main__":
    print(explain_anomaly("East", "P127", 11, 34, 5406.27))
