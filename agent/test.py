import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")

def send_slack_alert(message):
    if not SLACK_WEBHOOK:
        print("❌ Slack webhook not configured.")
        return

    try:
        response = requests.post(SLACK_WEBHOOK, json={"text": message})
        if response.status_code == 200:
            print("✅ Slack alert sent successfully!")
        else:
            print(f"❌ Slack error: {response.status_code}, {response.text}")
    except Exception as e:
        print("⚠️ Slack request failed:", e)

# Test message
send_slack_alert("🚨 This is a test message from LiveOps Agent!")
