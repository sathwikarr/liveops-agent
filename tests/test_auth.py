"""Tests for agent/auth.py — bcrypt signup + login."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def auth(tmp_db):
    """Reload auth module so it picks up the test DB fixture's monkeypatch."""
    import agent.auth as a
    importlib.reload(a)
    return a


def test_signup_creates_account(auth):
    ok, msg = auth.signup("alice", "correct-horse-battery")
    assert ok is True
    assert isinstance(msg, str)


def test_signup_rejects_duplicate(auth):
    ok1, _ = auth.signup("alice", "first-password-here")
    assert ok1 is True  # sanity: first signup actually succeeded
    ok, msg = auth.signup("alice", "different-password")
    assert ok is False
    assert "taken" in msg.lower() or "exist" in msg.lower()


def test_signup_rejects_short_password(auth):
    ok, msg = auth.signup("alice", "x")
    assert ok is False


def test_signup_rejects_invalid_username(auth):
    # Spaces, special characters — should fail the regex
    ok, msg = auth.signup("has space", "ok-password-here")
    assert ok is False
    ok2, _ = auth.signup("!!", "ok-password-here")
    assert ok2 is False


def test_login_correct_password(auth):
    auth.signup("alice", "pw-correct-horse")
    ok, msg = auth.login("alice", "pw-correct-horse")
    assert ok is True


def test_login_wrong_password(auth):
    auth.signup("alice", "pw-correct-horse")
    ok, _ = auth.login("alice", "wrong-pw")
    assert ok is False


def test_login_unknown_user(auth):
    ok, _ = auth.login("nobody", "anything")
    assert ok is False
