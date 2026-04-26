"""Action layer — rule-based decision with light reinforcement signal.

Persists actions and outcomes to SQLite (agent.db). Slack alerts are best-effort
and read SLACK_WEBHOOK at call time so .env loaded after import still applies.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests

from agent import db


def get_action_success_rate(username: Optional[str] = None) -> pd.Series:
    """Per-action success rate, optionally scoped to a single user."""
    return db.action_success_rates(username=username)


def simulate_action(
    username: Optional[str],
    region: str,
    product_id: str,
    orders: int,
    inventory: int,
    revenue: float,
) -> str:
    """Choose an action via rules + a crude RL gate on past success rate.

    Persists the chosen action with outcome='pending' and fires a Slack alert.
    Returns the action label.
    """
    success_rates = get_action_success_rate(username=username)

    if revenue > 1000 and inventory < 20:
        action = "🔁 Suggest rerouting inventory from other regions"
        if success_rates.get(action, 0) < 0.5:
            action = "📣 Boost ads or recommend item to more users"
    elif revenue > 1000 and orders > 15:
        action = "📣 Boost ads or recommend item to more users"
        if success_rates.get(action, 0) < 0.5:
            action = "🧪 Flag for pricing review or product audit"
    elif revenue > 1000:
        action = "🧪 Flag for pricing review or product audit"
    else:
        action = "ℹ️ Log only – no action needed"

    send_slack_alert(
        f"🚨 LiveOps Alert (user={username or 'system'}): {action} for {product_id} in {region} (rev={revenue})"
    )
    log_action(region, product_id, action, username=username)
    return action


def log_action(
    region: str,
    product_id: str,
    action: str,
    outcome: str = "pending",
    username: Optional[str] = None,
) -> int:
    """Persist an action; returns new row id."""
    return db.insert_action(
        username=username,
        region=region,
        product_id=product_id,
        action=action,
        outcome=outcome,
    )


def update_outcome(action_id: int, outcome: str) -> None:
    db.update_action_outcome(action_id, outcome)


def send_slack_alert(message: str) -> None:
    """Read SLACK_WEBHOOK at call time so .env loaded later still works."""
    webhook = os.getenv("SLACK_WEBHOOK")
    if not webhook:
        print("⚠️ Slack webhook not configured.")
        return
    try:
        requests.post(webhook, json={"text": message}, timeout=5)
    except Exception as e:
        print("Slack error:", e)
