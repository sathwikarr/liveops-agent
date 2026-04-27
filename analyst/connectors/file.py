"""FileConnector — wraps analyst.ingest so saved connections work for files
the same way they do for databases. Path is just a connection param."""
from __future__ import annotations

from pathlib import Path

from analyst.connectors.base import (
    Connector, ConnectionError, ConnectionResult,
)


class FileConnector(Connector):
    kind = "file"
    param_schema = {"path": "text"}

    def fetch(self) -> ConnectionResult:
        path = self.params.get("path")
        if not path:
            raise ConnectionError("Missing 'path' param")
        p = Path(path)
        if not p.exists():
            raise ConnectionError(f"File not found: {path}")
        # Lazy import — ingest is heavy and most tests don't need it
        from analyst.ingest import ingest
        try:
            res = ingest(str(p))
        except Exception as e:
            raise ConnectionError(f"Ingest failed: {e}") from e
        return ConnectionResult.from_df(
            res.df, source=str(p),
            kind=str(res.kind.kind) if hasattr(res.kind, "kind") else str(res.kind),
        )


__all__ = ["FileConnector"]
