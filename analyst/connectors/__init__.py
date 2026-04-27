"""Data connectors for the analyst workbench.

Each connector is a thin adapter that exposes a `fetch()` method returning
a pandas DataFrame. Heavy dependencies (psycopg2, gspread, boto3) are
imported lazily inside the relevant connector so the rest of the app
keeps working when those libraries are missing.

Saved connections live in SQLite (analyst/connectors/store.py) with
secrets encrypted using Fernet — the encryption key comes from the
ANALYST_CONN_KEY env var (auto-generated on first use if absent).
"""
from analyst.connectors.base import Connector, ConnectionError, ConnectionResult
from analyst.connectors.file import FileConnector
from analyst.connectors.postgres import PostgresConnector
from analyst.connectors.gsheets import GoogleSheetsConnector
from analyst.connectors.s3 import S3Connector
from analyst.connectors.store import (
    ConnectionStore, SavedConnection, REGISTRY,
)

__all__ = [
    "Connector", "ConnectionError", "ConnectionResult",
    "FileConnector", "PostgresConnector", "GoogleSheetsConnector",
    "S3Connector", "ConnectionStore", "SavedConnection", "REGISTRY",
]
