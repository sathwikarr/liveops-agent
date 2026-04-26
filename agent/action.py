from datetime import datetime
import pandas as pd
import os
import requests

from pathlib import Path

# Resolve relative to repo root so it works regardless of cwd
BASE_DIR = Path(__file__).resolve().parents[1]
ACTION_LOG = str(BASE_DIR / "data" / "action_log.csv")


def get_action_success_rate():
    if os.path.exists(ACTION_LOG):
        df = pd.read_csv(ACTION_LOG)
        if "outcome" not in df.columns or "action" not in df.columns:
            return pd.Series(dtype=float)
        success_df = df[df["outcome"] == "success"]
        if not success_df.empty:
            action_counts = success_df["action"].value_counts()
            total_successes = len(success_df)
            return action_counts / total_successes if total_successes > 0 else pd.Series(dtype=float)
    return pd.Series(dtype=float)


def simulate_action(username, region, product_id, orders, inventory, revenue):
    """
    Rule-based decision simulation with reinforcement learning based on past success rates.

    `username` is threaded through to the Slack alert and the action log so we
    know whose data triggered the action.
    """
    success_rates = get_action_success_rate()

    action = None
    if revenue > 1000 and inventory < 20:
        action = "🔁 Suggest rerouting inventory from other regions"
        if success_rates.get(action, 0) < 0.5:  # Switch if success rate < 50%
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
        f"🚨 LiveOps Alert (user={username}): {action} for {product_id} in {region} (rev={revenue})"
    )
    log_action(region, product_id, action, username=username)
    return action


def log_action(region, product_id, action, outcome="pending", username=None):
    Path(ACTION_LOG).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "username": username or "",
        "region": region,
        "product_id": product_id,
        "action": action,
        "outcome": outcome,
    }
    df = pd.DataFrame([data])
    df.to_csv(ACTION_LOG, mode="a", index=False, header=not os.path.exists(ACTION_LOG))


def send_slack_alert(message):
    """Read SLACK_WEBHOOK at call time so .env loaded later still takes effect."""
    webhook = os.getenv("SLACK_WEBHOOK")
    if not webhook:
        print("⚠️ Slack webhook not configured.")
        return
    try:
        requests.post(webhook, json={"text": message}, timeout=5)
    except Exception as e:
        print("Slack error:", e)
