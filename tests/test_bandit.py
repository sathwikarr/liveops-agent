"""Tests for agent/bandit.py — Thompson sampling action picker."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from agent import bandit


def test_pick_single_candidate(tmp_db):
    assert bandit.pick_action(["only-one"], username="u") == "only-one"


def test_pick_empty_candidates_raises(tmp_db):
    with pytest.raises(ValueError):
        bandit.pick_action([], username="u")


def test_arm_stats_zero_for_unseen(tmp_db):
    stats = bandit.arm_stats(["a", "b"], username="u")
    assert {s["action"] for s in stats} == {"a", "b"}
    for s in stats:
        assert s["successes"] == 0
        assert s["failures"] == 0
        assert s["mean"] is None


def test_arm_counts_reflect_db(tmp_db):
    # Seed actions
    for outcome in ["success", "success", "failed"]:
        tmp_db.insert_action(username="u", region="E", product_id="P1",
                             action="boost", outcome=outcome)
    s, f = bandit._arm_counts("boost", "u")
    assert s == 2
    assert f == 1


def test_pending_outcomes_dont_count(tmp_db):
    for outcome in ["pending", "ignored", "success"]:
        tmp_db.insert_action(username="u", region="E", product_id="P1",
                             action="boost", outcome=outcome)
    s, f = bandit._arm_counts("boost", "u")
    assert s == 1
    assert f == 0


def test_thompson_converges_to_better_arm(tmp_db):
    """Seed a clearly better arm and verify the bandit picks it most of the time."""
    # Arm A: 18/20 success
    for _ in range(18):
        tmp_db.insert_action(username="u", region="E", product_id="P1",
                             action="A", outcome="success")
    for _ in range(2):
        tmp_db.insert_action(username="u", region="E", product_id="P1",
                             action="A", outcome="failed")
    # Arm B: 4/20 success
    for _ in range(4):
        tmp_db.insert_action(username="u", region="E", product_id="P1",
                             action="B", outcome="success")
    for _ in range(16):
        tmp_db.insert_action(username="u", region="E", product_id="P1",
                             action="B", outcome="failed")

    rng = random.Random(0)
    picks = Counter(
        bandit.pick_action(["A", "B"], username="u", rng=rng) for _ in range(200)
    )
    # With seeded data this lopsided, A should win the vast majority of trials.
    assert picks["A"] > picks["B"] * 5


def test_per_user_isolation(tmp_db):
    """Action stats are scoped per user — Alice's failures don't penalize Bob."""
    for _ in range(10):
        tmp_db.insert_action(username="alice", region="E", product_id="P1",
                             action="X", outcome="failed")
    s_alice, f_alice = bandit._arm_counts("X", "alice")
    s_bob, f_bob = bandit._arm_counts("X", "bob")
    assert (s_alice, f_alice) == (0, 10)
    assert (s_bob, f_bob) == (0, 0)
