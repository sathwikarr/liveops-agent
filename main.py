from agent.detect import read_latest_data, zscore_anomaly_detection
from agent.explain import explain_anomaly
from agent.memory import save_anomaly_log
from agent.action import simulate_action
import streamlit as st
import os

REQUIRED_COLS = {"region", "product_id", "orders", "inventory", "revenue"}

def run_liveops_agent(username: str):
    st.write("🧠 LiveOps Agent Running...")
    user_csv_path = f"user_data/{username}.csv"

    # Check file
    if not os.path.exists(user_csv_path):
        st.warning("⚠️ No user CSV found. Please upload a file.")
        return

    # Load data (expects your read_latest_data(csv_path, n=...) to exist)
    df = read_latest_data(user_csv_path, n=500) if "read_latest_data" in globals() else None
    if df is None or df.empty:
        st.info("ℹ️ No rows to analyze yet.")
        return

    # Validate schema
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        return

    # Detect anomalies
    anomalies = zscore_anomaly_detection(df, "revenue", threshold=1.5)
    if anomalies.empty:
        st.success("✅ No anomalies found.")
        return

    st.subheader("⚠️ Detected Revenue Anomalies")
    for _, row in anomalies.iterrows():
        region = row["region"]
        product_id = row["product_id"]
        revenue = float(row["revenue"])

        # Pull latest context for that product+region safely
        latest = df[(df["product_id"] == product_id) & (df["region"] == region)].tail(1)
        if latest.empty:
            # Fallback: use the anomaly row if no latest match
            orders = int(row.get("orders", 0))
            inventory = int(row.get("inventory", 0))
        else:
            orders = int(latest["orders"].iloc[0])
            inventory = int(latest["inventory"].iloc[0])

        # Gemini-powered explanation
        explanation = explain_anomaly(region, product_id, orders, inventory, revenue)

        with st.expander(f"🚨 {product_id} in {region} — Revenue: {revenue:.2f}", expanded=False):
            st.markdown(explanation)

        # Persist + act
        try:
            save_anomaly(username, region, product_id, orders, inventory, revenue, explanation)
        except Exception as e:
            st.warning(f"⚠️ Could not save anomaly: {e}")

        try:
            action = simulate_action(username, region, product_id, orders, inventory, revenue)
            save_action(username, region, product_id, action)
            st.write("📦 Action Taken:", action)
        except Exception as e:
            st.warning(f"⚠️ Action/logging failed: {e}")

if __name__ == "__main__":
    if "username" in st.session_state:
        run_liveops_agent(st.session_state["username"])
    else:
        st.warning("⛔ Please log in to continue.")
        try:
            st.switch_page("app.py")
        except Exception:
            st.stop()
