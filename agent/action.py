from datetime import datetime
import pandas as pd
import os
import requests

ACTION_LOG = "data/action_log.csv"

def simulate_action(region, product_id, orders, inventory, revenue):
    """
    Simple rule-based decision simulation.
    Later this can use GPT or a business policy engine.
    """
    action = None

    if revenue > 1000 and inventory < 20:
        action = "🔁 Suggest rerouting inventory from other regions"
    elif revenue > 1000 and orders > 15:
        action = "📣 Boost ads or recommend item to more users"
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
        "outcome": outcome  # new!
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