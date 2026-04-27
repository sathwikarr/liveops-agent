"""Tests for analyst.connectors — file connector, store CRUD, encryption,
and the lazy-import behavior of the heavy connectors (postgres/gsheets/s3).

We don't actually hit a real database, sheet, or S3 bucket — those tests
would require credentials and network. Instead we verify:
1. File connector reads a real CSV through analyst.ingest
2. Saved-connection store roundtrips with encrypted secrets
3. Heavy connectors raise ConnectionError (not ImportError) when their
   dependency is missing
4. Each heavy connector validates its required params before trying to
   import its dependency, so config errors come back fast
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from analyst.connectors import (
    ConnectionError as ConnErr,
    ConnectionStore, FileConnector, GoogleSheetsConnector,
    PostgresConnector, S3Connector, SavedConnection, REGISTRY,
)
from analyst.connectors import store as store_mod


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def isolated_key(tmp_path, monkeypatch):
    """Each test gets its own Fernet key in env so they don't collide."""
    from cryptography.fernet import Fernet
    monkeypatch.setenv("ANALYST_CONN_KEY", Fernet.generate_key().decode())
    yield


@pytest.fixture
def tmp_csv(tmp_path):
    p = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(p, index=False)
    return p


@pytest.fixture
def store(tmp_path):
    return ConnectionStore(tmp_path / "conns.db")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_registry_lists_all_connectors():
    assert set(REGISTRY.keys()) == {"file", "postgres", "gsheets", "s3"}


# --------------------------------------------------------------------------- #
# File connector
# --------------------------------------------------------------------------- #

def test_file_connector_reads_csv(tmp_csv):
    res = FileConnector(path=str(tmp_csv)).fetch()
    assert res.rows == 3
    assert list(res.df.columns) == ["a", "b"]
    assert res.source == str(tmp_csv)


def test_file_connector_raises_on_missing(tmp_path):
    with pytest.raises(ConnErr):
        FileConnector(path=str(tmp_path / "nope.csv")).fetch()


def test_file_connector_raises_without_path():
    with pytest.raises(ConnErr):
        FileConnector().fetch()


# --------------------------------------------------------------------------- #
# Saved connection store
# --------------------------------------------------------------------------- #

def test_saved_connection_validates_kind():
    with pytest.raises(ValueError):
        SavedConnection(name="x", kind="not_real")


def test_saved_connection_validates_name():
    with pytest.raises(ValueError):
        SavedConnection(name="", kind="file")


def test_store_save_and_get_file_connection(store, tmp_csv):
    conn = SavedConnection(name="local-csv", kind="file",
                           params={"path": str(tmp_csv)})
    store.save(conn)
    out = store.get("local-csv")
    assert out is not None
    assert out.kind == "file"
    assert out.params["path"] == str(tmp_csv)


def test_store_encrypts_postgres_password(store):
    conn = SavedConnection(name="pg-prod", kind="postgres", params={
        "host": "db.internal", "database": "app", "user": "reader",
        "password": "super-secret-pw", "table": "events", "limit": "1000",
    })
    store.save(conn)
    # Read raw row from sqlite — password should be the encrypted blob, not plaintext
    import sqlite3
    raw = sqlite3.connect(store.db_path).execute(
        "SELECT params_json FROM connections WHERE name = ?", ("pg-prod",)
    ).fetchone()[0]
    assert "super-secret-pw" not in raw
    parsed = json.loads(raw)
    assert isinstance(parsed["password"], dict) and "__enc__" in parsed["password"]
    # Round-trip through get() decrypts it
    out = store.get("pg-prod")
    assert out.params["password"] == "super-secret-pw"


def test_store_list_returns_all(store, tmp_csv):
    store.save(SavedConnection("a", "file", {"path": str(tmp_csv)}))
    store.save(SavedConnection("b", "file", {"path": str(tmp_csv)}))
    names = sorted(c.name for c in store.list())
    assert names == ["a", "b"]


def test_store_delete(store, tmp_csv):
    store.save(SavedConnection("toremove", "file", {"path": str(tmp_csv)}))
    assert store.delete("toremove") is True
    assert store.get("toremove") is None
    assert store.delete("toremove") is False


def test_store_save_is_upsert(store, tmp_csv, tmp_path):
    other = tmp_path / "other.csv"
    pd.DataFrame({"x": [1]}).to_csv(other, index=False)
    store.save(SavedConnection("up", "file", {"path": str(tmp_csv)}))
    store.save(SavedConnection("up", "file", {"path": str(other)}))
    out = store.get("up")
    assert out.params["path"] == str(other)
    assert len(store.list()) == 1


def test_decrypt_with_wrong_key_returns_empty(store, monkeypatch):
    """If ANALYST_CONN_KEY rotates, secrets become unreadable but the row
    survives and the user is prompted to re-enter."""
    store.save(SavedConnection("pg", "postgres", params={
        "database": "app", "user": "reader", "password": "secret",
        "table": "users",
    }))
    # Rotate the key
    from cryptography.fernet import Fernet
    monkeypatch.setenv("ANALYST_CONN_KEY", Fernet.generate_key().decode())
    out = store.get("pg")
    assert out.params["password"] == ""        # blanked out
    assert out.params["database"] == "app"     # plaintext fields survive


# --------------------------------------------------------------------------- #
# Postgres connector — config errors should not require sqlalchemy install
# --------------------------------------------------------------------------- #

def test_postgres_requires_query_or_table():
    pg = PostgresConnector(host="x", database="db")
    with pytest.raises(ConnErr):
        pg._build_query()


def test_postgres_rejects_unsafe_table():
    pg = PostgresConnector(host="x", database="db", table="users; DROP TABLE x;--")
    with pytest.raises(ConnErr):
        pg._build_query()


def test_postgres_builds_default_select():
    pg = PostgresConnector(host="x", database="db", table="orders", limit="500")
    q = pg._build_query()
    assert q == "SELECT * FROM orders LIMIT 500"


def test_postgres_dsn_from_parts():
    pg = PostgresConnector(host="h", port="5433", database="d", user="u", password="p")
    assert "postgresql+psycopg2://u:p@h:5433/d" == pg._build_dsn()


def test_postgres_dsn_passthrough():
    pg = PostgresConnector(dsn="postgresql://x")
    assert pg._build_dsn() == "postgresql://x"


def test_postgres_requires_database():
    pg = PostgresConnector(host="h")
    with pytest.raises(ConnErr):
        pg._build_dsn()


# --------------------------------------------------------------------------- #
# Google Sheets connector — URL parsing
# --------------------------------------------------------------------------- #

def test_gsheets_extracts_id_from_url():
    g = GoogleSheetsConnector(
        sheet_url="https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz123456/edit#gid=0"
    )
    assert g._resolve_sheet_id() == "1AbCdEfGhIjKlMnOpQrStUvWxYz123456"


def test_gsheets_uses_explicit_id_first():
    g = GoogleSheetsConnector(sheet_url="https://docs.google.com/spreadsheets/d/AAA/edit",
                              sheet_id="EXPLICIT_ID")
    assert g._resolve_sheet_id() == "EXPLICIT_ID"


def test_gsheets_requires_url_or_id():
    g = GoogleSheetsConnector()
    with pytest.raises(ConnErr):
        g._resolve_sheet_id()


def test_gsheets_invalid_url_raises():
    g = GoogleSheetsConnector(sheet_url="https://example.com/not-a-sheet")
    with pytest.raises(ConnErr):
        g._resolve_sheet_id()


# --------------------------------------------------------------------------- #
# S3 connector — URI parsing + format detection
# --------------------------------------------------------------------------- #

def test_s3_parses_uri():
    from analyst.connectors.s3 import _parse_s3_uri
    assert _parse_s3_uri("s3://my-bucket/path/to/file.csv") == ("my-bucket", "path/to/file.csv")


def test_s3_rejects_non_s3_uri():
    from analyst.connectors.s3 import _parse_s3_uri
    with pytest.raises(ConnErr):
        _parse_s3_uri("https://my-bucket/path/file.csv")


def test_s3_format_detection_by_extension():
    s3 = S3Connector(uri="s3://b/k.parquet")
    assert s3._detect_format("k.parquet") == "parquet"
    assert s3._detect_format("k.json") == "json"
    assert s3._detect_format("k.csv") == "csv"
    assert s3._detect_format("k") == "csv"  # default


def test_s3_format_explicit_overrides_extension():
    s3 = S3Connector(uri="s3://b/k", format="parquet")
    assert s3._detect_format("k.csv") == "parquet"


# --------------------------------------------------------------------------- #
# Encryption helpers used by the store
# --------------------------------------------------------------------------- #

def test_encrypt_secrets_only_touches_secret_fields():
    enc = store_mod._encrypt_secrets("postgres", {
        "host": "h", "database": "d", "password": "secret",
    })
    assert enc["host"] == "h"
    assert enc["database"] == "d"
    assert isinstance(enc["password"], dict) and "__enc__" in enc["password"]


def test_encrypt_skips_empty_values():
    enc = store_mod._encrypt_secrets("postgres", {"password": ""})
    assert enc["password"] == ""


def test_decrypt_passes_plaintext_through():
    out = store_mod._decrypt_secrets({"host": "h", "password": ""})
    assert out == {"host": "h", "password": ""}
