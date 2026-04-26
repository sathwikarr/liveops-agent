"""LiveOps agent runner.

Imported by `pages/dashboard.py` so the "▶️ Run Agent Once" button can fire
the full detect → explain → log → act pipeline against the logged-in user's
uploaded CSV.

Lives in `pages/` so Streamlit's multipage discovery picks it up alongside
the dashboard, and so the dashboard's existing
`from pages.run_agent import run_liveops_agent` import resolves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make the repo root importable when Streamlit auto-runs this as a page
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from agent.detect import read_latest_data, zscore_anomaly_detection
from agent.explain import explain_anomaly
from agent.memory import save_anomaly_log
from agent.action import simulate_action

REQUIRED_COLS = {"region", "product_id", "orders", "inventory", "revenue"}


def run_liveops_agent(username: str) -> None:
    """Run one pass of the LiveOps pipeline for `username`.

    - Reads the user's uploaded CSV at `user_data/{username}.csv`
    - Runs z-score anomaly detection on revenue
    - For each anomaly: asks Gemini for an explanation, persists the anomaly,
      and triggers the rule-based action layer (which also fires a Slack alert
      and writes to `data/action_log.csv`).
    """
    st.write("🧠 LiveOps Agent Running...")

    user_csv_path = REPO_ROOT / "user_data" / f"{username}.csv"
    if not user_csv_path.exists():
        st.warning("⚠️ No user CSV found. Please upload a file.")
        return

    df = read_latest_data(str(user_csv_path), n=500)
    if df is None or df.empty:
        st.info("ℹ️ No rows to analyze yet.")
        return

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        return

    anomalies = zscore_anomaly_detection(df, "revenue", threshold=1.5)
    if anomalies.empty:
        st.success("✅ No anomalies found.")
        return

    st.subheader("⚠️ Detected Revenue Anomalies")
    for _, row in anomalies.iterrows():
        region = row["region"]
        product_id = row["product_id"]
        revenue = float(row["revenue"])

        latest = df[(df["product_id"] == product_id) & (df["region"] == region)].tail(1)
        if latest.empty:
            orders = int(row.get("orders", 0)) if "orders" in row else 0
            inventory = int(row.get("inventory", 0)) if "inventory" in row else 0
        else:
            orders = int(latest["orders"].iloc[0])
            inventory = int(latest["inventory"].iloc[0])

        explanation = explain_anomaly(region, product_id, orders, inventory, revenue)

        with st.expander(f"🚨 {product_id} in {region} — Revenue: {revenue:.2f}", expanded=False):
            st.markdown(explanation)

        try:
            save_anomaly_log(region, product_id, orders, inventory, revenue, explanation)
        except Exception as e:
            st.warning(f"⚠️ Could not save anomaly: {e}")

        try:
            action = simulate_action(username, region, product_id, orders, inventory, revenue)
            st.write("📦 Action Taken:", action)
        except Exception as e:
            st.warning(f"⚠️ Action/logging failed: {e}")


# When Streamlit auto-loads this as a page, render a stub instead of crashing
if __name__ == "__main__" or "streamlit" in os.path.basename(sys.argv[0] if sys.argv else ""):
    st.set_page_config(page_title="Run Agent")
    st.title("▶️ Run LiveOps Agent")
    if "username" not in st.session_state:
        st.warning("⛔ Please log in first.")
    else:
        run_liveops_agent(st.session_state["username"])
