"""Competitor context — optional layer that pulls public pricing/trend signals.

Sandbox-friendly: the actual web fetch is gated behind an `online=True` flag.
By default this module returns a deterministic stub so tests + offline use
work without network calls.

Public API:
- `lookup(product_names, online=False)` -> DataFrame[product, source, signal, value, note]
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def _stub(products: Iterable[str]) -> pd.DataFrame:
    rows = []
    for p in products:
        rows.append({"product": str(p), "source": "stub",
                     "signal": "competitor_price", "value": None,
                     "note": "Web lookup disabled — connect a search MCP to enable."})
    return pd.DataFrame(rows)


def lookup(products: Iterable[str], online: bool = False,
           fetch_fn=None) -> pd.DataFrame:
    """Look up competitor signals for a list of product names.

    `fetch_fn(product_name) -> dict` is called per product when `online=True`.
    Plug in any callable here (Claude in Chrome, an MCP tool, a custom
    scraper) — the module itself stays free of hard network dependencies.
    """
    products = list(products)
    if not products:
        return pd.DataFrame()
    if not online or fetch_fn is None:
        return _stub(products)

    rows = []
    for p in products:
        try:
            data = fetch_fn(p) or {}
            for sig, val in data.items():
                rows.append({"product": str(p), "source": "fetch",
                             "signal": sig, "value": val, "note": ""})
        except Exception as e:
            rows.append({"product": str(p), "source": "fetch",
                         "signal": "error", "value": None,
                         "note": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)


__all__ = ["lookup"]
