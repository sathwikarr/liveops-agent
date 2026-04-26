"""Action layer.

Pipeline per anomaly:
1. Build the rule-based candidate set (the original branching logic).
2. Let the Thompson-sampling bandit pick one based on past outcomes
   for THIS user.
3. Persist the action to SQLite.
4. Fan out a notification (Slack, optionally email for high+ severity).

Public API kept stable for callers (run_pipeline / dashboard):
- simulate_action(username, region, product_id, orders, inventory, revenue)
  -> str (action label)
- log_action(...)        -> int (row id)
- update_outcome(id, ..) -> None
- get_action_success_rate(username=None) -> pd.Series
- send_slack_alert(...)  -> compatibility shim, prefer notify()
"""
from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import requests

from agent import bandit, db
from agent.notify import notify

# Action labels are used as bandit arm IDs — keep stable.
ACT_REROUTE = "🔁 Suggest rerouting inventory from other regions"
ACT_BOOST = "📣 Boost ads or recommend item to more users"
ACT_AUDIT = "🧪 Flag for pricing review or product audit"
ACT_LOG_ONLY = "ℹ️ Log only – no action needed"


def get_action_success_rate(username: Optional[str] = None) -> pd.Series:
    return db.action_success_rates(username=username)


def _candidates(orders: int, inventory: int, revenue: float) -> List[str]:
    """Rule-based shortlist of plausible actions for the situation."""
    if revenue > 1000 and inventory < 20:
        return [ACT_REROUTE, ACT_BOOST]
    if revenue > 1000 and orders > 15:
        return [ACT_BOOST, ACT_AUDIT]
    if revenue > 1000:
        return [ACT_AUDIT, ACT_BOOST]
    return [ACT_LOG_ONLY]


def simulate_action(
    username: Optional[str],
    region: str,
    product_id: str,
    orders: int,
    inventory: int,
    revenue: float,
) -> str:
    """Pick + log + notify. Returns the chosen action label."""
    cands = _candidates(orders, inventory, revenue)
    action = bandit.pick_action(cands, username=username)

    # Optional structured explanation for severity. Imported lazily to avoid
    # the SDK import cost on cold paths and to keep this layer test-friendly.
    severity = "medium"
    rec = ""
    cause = ""
    try:
        from agent.explain import explain_anomaly_structured
        data = explain_anomaly_structured(region, product_id, orders, inventory, revenue)
        severity = data.get("severity", "medium")
        rec = data.get("recommended_action", "")
        cause = data.get("cause", "")
    except Exception:
        pass

    log_action(region, product_id, action, username=username)

    body = (
        f"Region: {region}\n"
        f"Product: {product_id}\n"
        f"Orders: {orders}  Inventory: {inventory}  Revenue: {revenue:.2f}\n"
        f"User: {username or 'system'}\n\n"
        f"Cause: {cause}\n"
        f"Recommended: {rec}\n"
        f"Selected action: {action}"
    )
    # Dedupe key: same (user, region, product) combo won't notify twice
    # within the cooldown window (default 30 min, ALERT_COOLDOWN_SECONDS env).
    dedupe_key = f"{username or 'system'}:{region}:{product_id}"
    try:
        notify(
            subject=f"{action} — {product_id} in {region}",
            body=body,
            severity=severity,
            dedupe_key=dedupe_key,
        )
    except Exception as e:
        print(f"[action] notify failed: {e}")

    return action


def log_action(
    region: str,
    product_id: str,
    action: str,
    outcome: str = "pending",
    username: Optional[str] = None,
) -> int:
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
    """Compat shim: routes through notify() at low severity."""
    try:
        notify(subject="LiveOps alert", body=message, severity="low")
    except Exception:
        webhook = os.getenv("SLACK_WEBHOOK")
        if not webhook:
            print("⚠️ Slack webhook not configured.")
            return
        try:
            requests.post(webhook, json={"text": message}, timeout=5)
        except Exception as e:
            print("Slack error:", e)
