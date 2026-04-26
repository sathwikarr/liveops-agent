"""Shared pytest fixtures.

Each test gets an isolated SQLite file via the LIVEOPS_DB env var so tests
don't pollute the developer's real database. We reload `agent.db` after
setting the env so the module-level DB_PATH picks up the override.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tmp_db(monkeypatch):
    """Point LIVEOPS_DB at a throwaway sqlite file and yield the reloaded module."""
    tmpdir = tempfile.mkdtemp(prefix="liveops-test-")
    db_path = os.path.join(tmpdir, "test.sqlite3")
    monkeypatch.setenv("LIVEOPS_DB", db_path)
    # Disable real channels so tests don't make network calls
    monkeypatch.setenv("SLACK_WEBHOOK", "")
    monkeypatch.setenv("SMTP_HOST", "")

    import agent.db as db
    importlib.reload(db)
    db.init_db()
    yield db
