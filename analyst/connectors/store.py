"""Saved-connection store — SQLite table that stores connector configs with
secrets encrypted via Fernet.

Encryption key resolution:
1. ANALYST_CONN_KEY env var (preferred — survives DB resets)
2. ~/.analyst_conn_key file (auto-generated on first use)

A SavedConnection round-trips through a single dict on the wire; secret
fields are encrypted in place. The REGISTRY maps `kind -> Connector class`
so the UI doesn't need to know about each connector individually.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from cryptography.fernet import Fernet, InvalidToken

from analyst.connectors.base import Connector
from analyst.connectors.file import FileConnector
from analyst.connectors.gsheets import GoogleSheetsConnector
from analyst.connectors.postgres import PostgresConnector
from analyst.connectors.s3 import S3Connector


REGISTRY: Dict[str, Type[Connector]] = {
    "file": FileConnector,
    "postgres": PostgresConnector,
    "gsheets": GoogleSheetsConnector,
    "s3": S3Connector,
}


# --------------------------------------------------------------------------- #
# Key management
# --------------------------------------------------------------------------- #

def _key_file_path() -> Path:
    return Path.home() / ".analyst_conn_key"


def _resolve_key() -> bytes:
    env_key = os.environ.get("ANALYST_CONN_KEY")
    if env_key:
        return env_key.encode() if isinstance(env_key, str) else env_key
    p = _key_file_path()
    if p.exists():
        return p.read_bytes().strip()
    # First run — generate and persist
    k = Fernet.generate_key()
    try:
        p.write_bytes(k)
        p.chmod(0o600)
    except Exception:
        # Read-only filesystem (eg. some serverless hosts) — fall back to
        # in-memory only; secrets won't survive a restart but won't crash.
        pass
    return k


def _fernet() -> Fernet:
    return Fernet(_resolve_key())


# --------------------------------------------------------------------------- #
# Saved connection model
# --------------------------------------------------------------------------- #

@dataclass
class SavedConnection:
    name: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if self.kind not in REGISTRY:
            raise ValueError(f"Unknown connector kind: {self.kind}")
        if not self.name or not self.name.strip():
            raise ValueError("Saved connection needs a non-empty name")
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat(timespec="seconds")
        if not self.updated_at:
            self.updated_at = self.created_at

    def connector(self) -> Connector:
        cls = REGISTRY[self.kind]
        return cls(**self.params)


# --------------------------------------------------------------------------- #
# Encrypt / decrypt helpers
# --------------------------------------------------------------------------- #

def _encrypt_secrets(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    schema = REGISTRY[kind].param_schema
    f = _fernet()
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None or v == "":
            out[k] = v
            continue
        if schema.get(k) == "secret":
            token = f.encrypt(str(v).encode()).decode()
            out[k] = {"__enc__": token}
        else:
            out[k] = v
    return out


def _decrypt_secrets(params: Dict[str, Any]) -> Dict[str, Any]:
    f = _fernet()
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, dict) and "__enc__" in v:
            try:
                out[k] = f.decrypt(v["__enc__"].encode()).decode()
            except InvalidToken:
                # Key changed — return placeholder so the UI prompts the user
                out[k] = ""
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Store — SQLite-backed
# --------------------------------------------------------------------------- #

class ConnectionStore:
    """Tiny CRUD wrapper around a single SQLite table."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS connections (
                    name TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, conn: SavedConnection) -> None:
        params_enc = _encrypt_secrets(conn.kind, conn.params)
        conn.updated_at = datetime.utcnow().isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO connections (name, kind, params_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    kind = excluded.kind,
                    params_json = excluded.params_json,
                    updated_at = excluded.updated_at
                """,
                (conn.name, conn.kind, json.dumps(params_enc),
                 conn.created_at, conn.updated_at),
            )

    def get(self, name: str) -> Optional[SavedConnection]:
        with self._conn() as c:
            row = c.execute(
                "SELECT name, kind, params_json, created_at, updated_at "
                "FROM connections WHERE name = ?", (name,)
            ).fetchone()
        if not row:
            return None
        params_enc = json.loads(row["params_json"])
        params = _decrypt_secrets(params_enc)
        return SavedConnection(
            name=row["name"], kind=row["kind"], params=params,
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def list(self) -> List[SavedConnection]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT name, kind, params_json, created_at, updated_at "
                "FROM connections ORDER BY updated_at DESC"
            ).fetchall()
        out: List[SavedConnection] = []
        for r in rows:
            params = _decrypt_secrets(json.loads(r["params_json"]))
            out.append(SavedConnection(
                name=r["name"], kind=r["kind"], params=params,
                created_at=r["created_at"], updated_at=r["updated_at"],
            ))
        return out

    def delete(self, name: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM connections WHERE name = ?", (name,))
            return cur.rowcount > 0


__all__ = ["ConnectionStore", "SavedConnection", "REGISTRY",
           "_encrypt_secrets", "_decrypt_secrets"]
