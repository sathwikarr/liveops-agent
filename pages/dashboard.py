import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------- Page Setup -----------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 LiveOps Agent - Dashboard")
st.caption("Real-time anomaly detection + explanations (Gemini-powered)")

# Add repo root to path for imports (works from pages/)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from agent.detect import read_latest_data, zscore_anomaly_detection
from agent.explain import explain_anomaly
from agent.memory import read_anomaly_log
from agent.action import get_action_success_rate
from agent.forecast import forecast_revenue

# Optional: import your runner to trigger logging from the dashboard
try:
    from pages.run_agent import run_liveops_agent  # adjust if your runner lives elsewhere
    HAS_RUNNER = True
except Exception:
    HAS_RUNNER = False

REQUIRED_COLS = {"region", "product_id", "revenue", "orders", "inventory"}

# ------------------- Auth Check ----------------
if "username" not in st.session_state:
    st.warning("⛔ Please log in to continue.")
    try:
        st.switch_page("app.py")  # Streamlit v1.31+
    except Exception:
        st.stop()

username = st.session_state["username"]
st.sidebar.success(f"Logged in as: {username}")
st.sidebar.button("🚪 Log out", on_click=lambda: st.session_state.clear())

# Controls
st.sidebar.markdown("### Settings")
top_k = st.sidebar.number_input("Explain top-K anomalies per run", min_value=1, max_value=50, value=5, step=1)
z_thresh = st.sidebar.slider("Z-score threshold", 0.5, 4.0, 1.5, 0.1)

# ------------------- File Upload ----------------
st.subheader("📤 Upload Your CSV")
user_csv_path = REPO_ROOT / "user_data" / f"{username}.csv"
uploaded = st.file_uploader("Upload your ops CSV", type="csv")

if uploaded:
    user_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(user_csv_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("✅ File uploaded! Please refresh to see results.")

if not user_csv_path.exists():
    st.warning("⚠️ No CSV uploaded yet. Please upload your file to continue.")
    st.stop()

# ----------------- Live Data -----------------
try:
    df = pd.read_csv(user_csv_path)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("📈 Live Data Stream")
if "timestamp" in df.columns:
    st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
else:
    st.dataframe(df, use_container_width=True)

# ----------------- Anomaly Detection -----------------
if REQUIRED_COLS.issubset(df.columns):
    anomalies = zscore_anomaly_detection(df, "revenue", threshold=float(z_thresh))

    if anomalies.empty:
        st.success("✅ No anomalies detected.")
    else:
        # Sort by strongest deviation first
        if "z_score" in anomalies.columns:
            anomalies = anomalies.sort_values("z_score", key=lambda s: s.abs(), ascending=False)

        st.subheader("⚠️ Detected Revenue Anomalies")
        # Limit how many we explain this render
        explain_subset = anomalies.head(int(top_k))
        omitted = len(anomalies) - len(explain_subset)

        @st.cache_data(ttl=1800, show_spinner=False)
        def cached_explain(region, product_id, orders, inventory, revenue):
            # cache key is the full tuple; keep types stable
            return explain_anomaly(region, product_id, int(orders), int(inventory), float(revenue))

        for _, row in explain_subset.iterrows():
            region = row["region"]
            product_id = row["product_id"]
            revenue = float(row["revenue"])

            # Safely get the latest row for this product/region
            latest = df[(df["product_id"] == product_id) & (df["region"] == region)]
            if latest.empty:
                orders = int(row.get("orders", 0)) if "orders" in row else 0
                inventory = int(row.get("inventory", 0)) if "inventory" in row else 0
            else:
                latest_row = latest.tail(1).iloc[0]
                orders = int(latest_row.get("orders", 0))
                inventory = int(latest_row.get("inventory", 0))

            explanation = cached_explain(region, product_id, orders, inventory, revenue)

            title = f"🚨 {product_id} in {region} — Revenue: {revenue:,.2f}"
            if "z_score" in anomalies.columns:
                title += f" (z={row['z_score']:.2f})"

            with st.expander(title):
                st.markdown(explanation)

        if omitted > 0:
            st.info(f"Showing explanations for top **{len(explain_subset)}** anomalies. "
                    f"**{omitted}** more omitted to stay within API limits.")
else:
    st.warning("⚠️ CSV is missing required columns: region, product_id, orders, inventory, revenue")

# ----------------- Agent Log -----------------
from agent.memory import get_log_path  # show exact path for debugging
st.subheader("📜 Agent Log History")
st.caption(f"🔎 Reading anomaly log from: `{get_log_path()}`")

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("🔄 Refresh logs"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()
with col_b:
    if HAS_RUNNER and st.button("▶️ Run Agent Once"):
        # Run the agent once to generate/save logs (uses user's CSV)
        run_liveops_agent(username)
        st.rerun()

log_df = read_anomaly_log()
if log_df.empty:
    st.info("No logs yet.")
else:
    st.dataframe(log_df.sort_values(by="timestamp", ascending=False), use_container_width=True)

# ----------------- Action Log & Editable Outcomes -----------------
st.subheader("📦 Action Log & Outcomes")
ACTION_LOG_PATH = REPO_ROOT / "data" / "action_log.csv"  # absolute path

if ACTION_LOG_PATH.exists():
    try:
        action_df = pd.read_csv(ACTION_LOG_PATH)
        st.dataframe(action_df.sort_values(by="timestamp", ascending=False), use_container_width=True)

        st.info("Edit outcomes below and click Save to update:")
        editable_df = st.data_editor(action_df, num_rows="dynamic", key="action_log_editor")
        if st.button("💾 Save Updated Outcomes"):
            editable_df.to_csv(ACTION_LOG_PATH, index=False)
            st.success("Outcome updates saved.")
    except Exception as e:
        st.warning(f"Could not read or update action log: {e}")
else:
    st.info("No actions taken yet.")

# ----------------- Outcome Charts -----------------
st.subheader("📊 Agent Outcome Analytics")
if ACTION_LOG_PATH.exists():
    try:
        outcome_df = pd.read_csv(ACTION_LOG_PATH)
        if "outcome" in outcome_df.columns and not outcome_df.empty:
            outcome_counts = outcome_df["outcome"].value_counts()

            # Pie Chart
            fig1, ax1 = plt.subplots()
            ax1.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            # Bar Chart by Date
            if "timestamp" in outcome_df.columns:
                outcome_df["date"] = pd.to_datetime(outcome_df["timestamp"], errors="coerce").dt.date
                actions_by_day = outcome_df.groupby(["date", "outcome"]).size().unstack(fill_value=0)
                if not actions_by_day.empty:
                    st.bar_chart(actions_by_day)
    except Exception as e:
        st.warning(f"Could not render analytics: {e}")

# ----------------- Agent Learning Insights -----------------
st.subheader("🧠 Agent Learning Insights")
success_rates = get_action_success_rate()
if hasattr(success_rates, "empty") and not success_rates.empty:
    st.write("Action Success Rates:")
    for action, rate in success_rates.items():
        st.write(f"{action}: {rate:.2%} success rate")
else:
    st.info("No success data yet — mark outcomes to start learning!")

# ----------------- Revenue Forecast -----------------
st.subheader("🔮 Revenue Forecast")
try:
    forecast_df = forecast_revenue(user_csv_path, periods=10)
    if not forecast_df.empty:
        st.line_chart(forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

        anomalies_forecast = forecast_df[forecast_df["anomaly"]]
        if not anomalies_forecast.empty:
            st.warning(f"⚠️ Potential future anomalies detected: {len(anomalies_forecast)}")
            st.dataframe(anomalies_forecast[["ds", "yhat", "anomaly"]])
        else:
            st.success("✅ No significant anomalies forecasted.")
    else:
        st.info("No forecast data available yet.")
except Exception as e:
    st.info(f"Forecast not available ({e}).")
