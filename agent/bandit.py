"""Thompson sampling action selector.

Beta-Bernoulli bandit, one arm per (action_label, username). Pulls successes
and failures from db.actions and samples a posterior to pick which candidate
action to take next.

Why Thompson over the old `success_rate < 0.5` gate:
- Cold start: the gate fires arbitrarily before any data exists.
- No exploration: once any action's rate dips below 0.5 it's never tried again
  even if it was unlucky.
- Per-user bias: the gate aggregated globally, so one user's bad outcomes
  starved the action for everyone.

Beta(α, β) prior with α=β=1 (uniform) — switches to "explore" naturally when
counts are low and tightens to "exploit" as evidence accumulates.
"""
from __future__ import annotations

import random
from typing import Iterable, Optional

from agent import db


def _arm_counts(action: str, username: Optional[str]) -> tuple[int, int]:
    """Return (successes, failures) for an arm. Pending/ignored don't count."""
    with db.connect() as conn:
        if username:
            row = conn.execute(
                """SELECT
                       SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END) AS s,
                       SUM(CASE WHEN outcome='failed'  THEN 1 ELSE 0 END) AS f
                   FROM actions
                   WHERE action = ? AND username = ?""",
                (action, username),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT
                       SUM(CASE WHEN outcome='success' THEN 1 ELSE 0 END) AS s,
                       SUM(CASE WHEN outcome='failed'  THEN 1 ELSE 0 END) AS f
                   FROM actions
                   WHERE action = ?""",
                (action,),
            ).fetchone()
    s = int(row["s"] or 0) if row else 0
    f = int(row["f"] or 0) if row else 0
    return s, f


def sample_arm(action: str, username: Optional[str], rng: Optional[random.Random] = None) -> float:
    """Sample θ ~ Beta(α=1+s, β=1+f). Higher = better expected success."""
    s, f = _arm_counts(action, username)
    rng = rng or random
    return rng.betavariate(1 + s, 1 + f)


def pick_action(
    candidates: Iterable[str],
    username: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Thompson-sample each candidate; return the action with the highest draw."""
    cands = list(candidates)
    if not cands:
        raise ValueError("pick_action requires at least one candidate")
    if len(cands) == 1:
        return cands[0]
    return max(cands, key=lambda a: sample_arm(a, username, rng))


def arm_stats(candidates: Iterable[str], username: Optional[str] = None) -> list[dict]:
    """Return per-arm (action, successes, failures, mean) — useful for UI."""
    out = []
    for a in candidates:
        s, f = _arm_counts(a, username)
        n = s + f
        out.append({
            "action": a,
            "successes": s,
            "failures": f,
            "mean": (s / n) if n > 0 else None,
        })
    return out
