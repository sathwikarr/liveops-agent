"""Connector base class — every adapter implements `fetch()` and returns
a `ConnectionResult` with the DataFrame plus debug info."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


class ConnectionError(Exception):
    """Raised when a connector fails to retrieve data. Wraps the underlying
    cause so callers can surface a useful message without dragging in the
    upstream library's exception type."""


@dataclass
class ConnectionResult:
    df: pd.DataFrame
    source: str                       # human-readable origin (uri, sheet name, …)
    rows: int = 0
    columns: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_df(cls, df: pd.DataFrame, source: str,
                **extras: Any) -> "ConnectionResult":
        return cls(df=df, source=source,
                   rows=len(df), columns=len(df.columns), extras=dict(extras))


class Connector:
    """Abstract base — subclass and implement `fetch()`. The kind string is
    used for routing in the saved-connection store (postgres / gsheets / s3
    / file). Each concrete connector owns its own param schema."""

    kind: str = "abstract"
    param_schema: Dict[str, str] = {}     # name -> "secret" | "text" | "url"

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def fetch(self) -> ConnectionResult:  # pragma: no cover - subclass override
        raise NotImplementedError


__all__ = ["Connector", "ConnectionError", "ConnectionResult"]
