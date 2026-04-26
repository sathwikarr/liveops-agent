"""Tests for agent/db.py — schema, users, anomalies, actions, dedupe."""
from __future__ import annotations

import time


def test_init_idempotent(tmp_db):
    tmp_db.init_db()
    tmp_db.init_db()  # safe to call repeatedly


def test_create_user_and_get(tmp_db):
    assert tmp_db.create_user("alice", "hash-1") is True
    # Duplicate username returns False
    assert tmp_db.create_user("alice", "hash-2") is False

    row = tmp_db.get_user("alice")
    assert row is not None
    assert row["password_hash"] == "hash-1"

    assert "alice" in tmp_db.list_usernames()


def test_insert_and_read_anomaly(tmp_db):
    aid = tmp_db.insert_anomaly(
        username="alice", region="East", product_id="P127",
        orders=10, inventory=50, revenue=999.0,
        metric="revenue", z_score=3.5, explanation="spike",
    )
    assert isinstance(aid, int) and aid > 0

    df = tmp_db.read_anomalies(username="alice")
    assert len(df) == 1
    assert df.iloc[0]["region"] == "East"
    assert df.iloc[0]["product_id"] == "P127"
    assert df.iloc[0]["z_score"] == 3.5


def test_insert_and_read_action(tmp_db):
    aid = tmp_db.insert_action(
        username="alice", region="East", product_id="P127",
        action="restock", outcome="pending",
    )
    assert aid > 0
    df = tmp_db.read_actions(username="alice")
    assert len(df) == 1
    assert df.iloc[0]["outcome"] == "pending"

    # Update outcome
    tmp_db.update_action_outcome(aid, "success")
    df2 = tmp_db.read_actions(username="alice")
    assert df2.iloc[0]["outcome"] == "success"


def test_action_success_rates(tmp_db):
    for outcome in ["success", "success", "failed", "pending"]:
        aid = tmp_db.insert_action(username="alice", region="E", product_id="P1",
                                   action="boost", outcome=outcome)
    rates = tmp_db.action_success_rates(username="alice")
    # 2 successes / 3 decided = 0.666... — pending excluded
    assert "boost" in rates.index
    assert abs(rates["boost"] - 2 / 3) < 1e-9


def test_dedupe_within_window_is_blocked(tmp_db):
    key = "alice:East:P127"
    assert tmp_db.should_notify(key, cooldown_seconds=60) is True
    # Second call inside the window — blocked.
    assert tmp_db.should_notify(key, cooldown_seconds=60) is False


def test_dedupe_after_cooldown_fires_again(tmp_db):
    key = "alice:East:P127"
    assert tmp_db.should_notify(key, cooldown_seconds=1) is True
    time.sleep(1.2)
    # Window has elapsed — fires again.
    assert tmp_db.should_notify(key, cooldown_seconds=1) is True


def test_dedupe_different_keys_are_independent(tmp_db):
    assert tmp_db.should_notify("alice:East:P1", cooldown_seconds=60) is True
    assert tmp_db.should_notify("alice:East:P2", cooldown_seconds=60) is True
    assert tmp_db.should_notify("bob:East:P1", cooldown_seconds=60) is True


def test_dedupe_empty_key_always_fires(tmp_db):
    assert tmp_db.should_notify("", cooldown_seconds=60) is True
    assert tmp_db.should_notify("", cooldown_seconds=60) is True
