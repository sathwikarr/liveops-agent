"""PostgresConnector — runs a SQL query (or `SELECT * FROM table`) and
returns the result as a DataFrame.

Two ways to specify the connection:
1. `dsn`: full SQLAlchemy URL ("postgresql+psycopg2://user:pw@host:5432/db")
2. `host`/`port`/`database`/`user`/`password` (built into a DSN at runtime)

Either `query` (raw SQL) OR `table` (becomes "SELECT * FROM <table> LIMIT N")
is required. Row cap defaults to 100k to keep the UI responsive.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from analyst.connectors.base import (
    Connector, ConnectionError, ConnectionResult,
)


_SAFE_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.]{0,127}$")


class PostgresConnector(Connector):
    kind = "postgres"
    param_schema = {
        "dsn": "secret",
        "host": "text", "port": "text", "database": "text",
        "user": "text", "password": "secret",
        "query": "text", "table": "text", "limit": "text",
    }

    def _build_dsn(self) -> str:
        if self.params.get("dsn"):
            return str(self.params["dsn"])
        host = self.params.get("host", "localhost")
        port = self.params.get("port", "5432")
        db = self.params.get("database", "")
        user = self.params.get("user", "")
        pw = self.params.get("password", "")
        if not db:
            raise ConnectionError("database is required")
        creds = f"{user}:{pw}@" if user else ""
        return f"postgresql+psycopg2://{creds}{host}:{port}/{db}"

    def _build_query(self) -> str:
        q = (self.params.get("query") or "").strip()
        if q:
            return q
        table = (self.params.get("table") or "").strip()
        if not table:
            raise ConnectionError("Provide either `query` or `table`")
        if not _SAFE_TABLE.match(table):
            raise ConnectionError(f"Refusing to query unsafe identifier: {table}")
        try:
            limit = int(self.params.get("limit") or 100_000)
        except ValueError:
            limit = 100_000
        return f"SELECT * FROM {table} LIMIT {limit}"

    def fetch(self) -> ConnectionResult:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as e:
            raise ConnectionError(
                "sqlalchemy is required for the postgres connector — "
                "pip install sqlalchemy psycopg2-binary"
            ) from e

        dsn = self._build_dsn()
        query = self._build_query()
        try:
            engine = create_engine(dsn, pool_pre_ping=True)
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
        except Exception as e:
            raise ConnectionError(f"Postgres query failed: {e}") from e
        # Redact creds from the source string we surface to the UI
        safe_dsn = re.sub(r"(://)([^:@/]+):([^@/]*)@", r"\1\2:***@", dsn)
        return ConnectionResult.from_df(
            df, source=f"{safe_dsn} :: {query[:80]}",
            query=query,
        )


__all__ = ["PostgresConnector"]
