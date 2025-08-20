from datetime import datetime
import pandas as pd
import os
import requests

ACTION_LOG = "data/action_log.csv"

def get_action_success_rate():
    if os.path.exists(ACTION_LOG):
        df = pd.read_csv(ACTION_LOG)
        success_df = df[df['outcome'] == 'success']
        if not success_df.empty:
            action_counts = success_df['action'].value_counts()
            total_successes = len(success_df)
            return action_counts / total_successes if total_successes > 0 else pd.Series()
    return pd.Series()

def simulate_action(region, product_id, orders, inventory, revenue):
    """
    Rule-based decision simulation with reinforcement learning based on past success rates.
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

    send_slack_alert(f"🚨 LiveOps Alert: {action} for {product_id} in {region} (rev={revenue})")
    log_action(region, product_id, action)
    return action

def log_action(region, product_id, action, outcome="pending"):
    data = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "region": region,
        "product_id": product_id,
        "action": action,
        "outcome": outcome
    }
    df = pd.DataFrame([data])
    df.to_csv(ACTION_LOG, mode='a', index=False, header=not os.path.exists(ACTION_LOG))

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")

def send_slack_alert(message):
    if not SLACK_WEBHOOK:
        print("⚠️ Slack webhook not configured.")
        return

    try:
        requests.post(SLACK_WEBHOOK, json={"text": message})
    except Exception as e:
        print("Slack error:", e)