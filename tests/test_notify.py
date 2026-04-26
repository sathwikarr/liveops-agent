"""Tests for agent/notify.py — severity routing + dedupe."""
from __future__ import annotations

from agent.notify import notify, _at_least, SEVERITY_ORDER


def test_at_least_severity():
    assert _at_least("critical", "low") is True
    assert _at_least("low", "high") is False
    assert _at_least("medium", "medium") is True


def test_severity_order_complete():
    assert set(SEVERITY_ORDER.keys()) == {"low", "medium", "high", "critical"}


def test_notify_no_channels_configured(tmp_db, monkeypatch):
    """With SLACK_WEBHOOK and SMTP_HOST unset, both channels skip cleanly."""
    monkeypatch.delenv("SLACK_WEBHOOK", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)
    r = notify(subject="t", body="b", severity="medium")
    assert r["slack"] is False
    assert r["email"] is False
    assert r["skipped"] is False


def test_notify_dedupe_first_call_passes(tmp_db):
    r = notify(subject="t", body="b", severity="medium",
               dedupe_key="alice:E:P1", cooldown_seconds=60)
    assert r["skipped"] is False


def test_notify_dedupe_second_call_blocked(tmp_db):
    notify(subject="t", body="b", severity="medium",
           dedupe_key="alice:E:P1", cooldown_seconds=60)
    r2 = notify(subject="t", body="b", severity="medium",
                dedupe_key="alice:E:P1", cooldown_seconds=60)
    assert r2["skipped"] is True
    assert r2["slack"] is False
    assert r2["email"] is False


def test_notify_dedupe_different_keys_independent(tmp_db):
    r1 = notify(subject="t", body="b", severity="medium",
                dedupe_key="alice:E:P1", cooldown_seconds=60)
    r2 = notify(subject="t", body="b", severity="medium",
                dedupe_key="alice:E:P2", cooldown_seconds=60)
    assert r1["skipped"] is False
    assert r2["skipped"] is False


def test_notify_no_dedupe_key_always_passes(tmp_db):
    r1 = notify(subject="t", body="b", severity="medium")
    r2 = notify(subject="t", body="b", severity="medium")
    assert r1["skipped"] is False
    assert r2["skipped"] is False


def test_email_skipped_below_min_severity(tmp_db, monkeypatch):
    """ALERT_EMAIL_MIN_SEVERITY=critical means medium-severity events skip email."""
    monkeypatch.setenv("ALERT_EMAIL_MIN_SEVERITY", "critical")
    monkeypatch.setenv("SMTP_HOST", "")  # explicitly off
    r = notify(subject="t", body="b", severity="medium")
    assert r["email"] is False
