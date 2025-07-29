import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.detect import read_latest_data, zscore_anomaly_detection
from main import mock_explain
from agent.memory import read_anomaly_log

# ----------------- Page Setup -----------------
st.set_page_config(page_title="LiveOps Agent Dashboard", layout="wide")
st.title("📊 LiveOps Agent Dashboard")
st.caption("Real-time anomaly detection + explanations")

# ----------------- Live Data -----------------
df = read_latest_data()
st.subheader("📈 Live Data Stream")
st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)

# ----------------- Anomaly Detection -----------------
anomalies = zscore_anomaly_detection(df, "revenue")

if anomalies.empty:
    st.success("✅ No anomalies detected.")
else:
    st.subheader("⚠️ Detected Revenue Anomalies")
    for _, row in anomalies.iterrows():
        region = row["region"]
        product_id = row["product_id"]
        revenue = row["revenue"]
        orders = df[(df["product_id"] == product_id) & (df["region"] == region)]["orders"].iloc[-1]
        inventory = df[(df["product_id"] == product_id) & (df["region"] == region)]["inventory"].iloc[-1]

        explanation = mock_explain(region, product_id, orders, inventory, revenue)

        with st.expander(f"🚨 {product_id} in {region} — Revenue: {revenue}"):
            st.write(explanation)

# ----------------- Agent Log -----------------
st.subheader("📜 Agent Log History")
log_df = read_anomaly_log()

if log_df.empty:
    st.info("No logs yet.")
else:
    st.dataframe(log_df.sort_values(by="timestamp", ascending=False), use_container_width=True)

# ----------------- Action Log View -----------------
st.subheader("📦 Action Log")
action_log_path = "data/action_log.csv"

if os.path.exists(action_log_path):
    action_df = pd.read_csv(action_log_path)
    st.dataframe(action_df.sort_values(by="timestamp", ascending=False), use_container_width=True)
else:
    st.info("No actions taken yet.")

# ----------------- Editable Outcomes -----------------
st.subheader("📦 Action Log (Mark Outcomes)")

if os.path.exists("data/action_log.csv"):
    action_df = pd.read_csv("data/action_log.csv")

    st.info("You can edit outcomes directly below and click Save to update:")
    
    editable_df = st.data_editor(
        action_df,
        num_rows="dynamic",
        key="action_log_editor"
    )

    if st.button("💾 Save Updated Outcomes"):
        editable_df.to_csv("data/action_log.csv", index=False)
        st.success("Outcome updates saved.")
else:
    st.info("No actions taken yet.")

# ----------------- Outcome Charts -----------------
st.subheader("📊 Agent Outcome Analytics")

if os.path.exists(action_log_path):
    outcome_df = pd.read_csv(action_log_path)

    # 1. Pie Chart
    outcome_counts = outcome_df["outcome"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # 2. Bar Chart by Date
    if "timestamp" in outcome_df.columns:
        outcome_df["date"] = pd.to_datetime(outcome_df["timestamp"]).dt.date
        actions_by_day = outcome_df.groupby(["date", "outcome"]).size().unstack(fill_value=0)

        st.bar_chart(actions_by_day)
