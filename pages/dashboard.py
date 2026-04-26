"""LiveOps Dashboard — reads from SQLite, runs the agent, edits outcomes."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ----------------- Page setup -----------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 LiveOps Agent — Dashboard")
st.caption("Real-time anomaly detection + LLM-powered explanations (Gemini)")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from agent import db
from agent.action import get_action_success_rate, update_outcome
from agent.detect import detect_anomalies
from agent.forecast import forecast_revenue
from agent.memory import get_log_path
from pages.run_agent import run_liveops_agent

REQUIRED_COLS = {"region", "product_id", "revenue", "orders", "inventory"}

# ----------------- Auth check -----------------
if "username" not in st.session_state:
    st.warning("⛔ Please log in to continue.")
    try:
        st.switch_page("app.py")
    except Exception:
        st.stop()

username: str = st.session_state["username"]
st.sidebar.success(f"Logged in as: {username}")
st.sidebar.button("🚪 Log out", on_click=lambda: st.session_state.clear())

# ----------------- Sidebar settings -----------------
st.sidebar.markdown("### Settings")
top_k = st.sidebar.number_input("Explain top-K anomalies per run", 1, 50, 5, 1)
z_thresh = st.sidebar.slider("Z-score threshold", 0.5, 4.0, 2.0, 0.1)

# ----------------- File upload -----------------
st.subheader("📤 Upload Your CSV")
user_csv_path = REPO_ROOT / "user_data" / f"{username}.csv"
uploaded = st.file_uploader("Upload your ops CSV", type="csv")

if uploaded:
    user_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(user_csv_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("✅ File uploaded! Refreshing…")
    st.rerun()

if not user_csv_path.exists():
    st.warning("⚠️ No CSV uploaded yet. Please upload your file to continue.")
    st.stop()

# ----------------- Live data -----------------
try:
    df = pd.read_csv(user_csv_path)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("📈 Live Data Stream")
view = df.sort_values("timestamp", ascending=False) if "timestamp" in df.columns else df
st.dataframe(view, use_container_width=True)

# ----------------- Anomaly detection (per-segment, multi-metric) -----------------
if REQUIRED_COLS.issubset(df.columns):
    anomalies = detect_anomalies(df, threshold=float(z_thresh))

    if anomalies.empty:
        st.success("✅ No anomalies detected.")
    else:
        st.subheader("⚠️ Detected Anomalies")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(anomalies))
        c2.metric("Per-segment", int((anomalies["scope"] == "segment").sum()))
        c3.metric("Global", int((anomalies["scope"] == "global").sum()))

        st.dataframe(
            anomalies[["timestamp", "region", "product_id", "metric", "value", "z_score", "scope"]],
            use_container_width=True, height=240,
        )
else:
    st.warning("⚠️ CSV is missing required columns: region, product_id, orders, inventory, revenue")

# ----------------- Run agent + log history -----------------
st.subheader("📜 Agent Log History")
st.caption(f"🔎 Database: `{get_log_path()}`")

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("🔄 Refresh"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        st.rerun()
with col_b:
    if st.button("▶️ Run Agent Once"):
        run_liveops_agent(username)
        st.rerun()

anomaly_log = db.read_anomalies(username=username, limit=200)
if anomaly_log.empty:
    st.info("No anomalies logged yet.")
else:
    st.dataframe(anomaly_log.rename(columns={"ts": "timestamp"}), use_container_width=True)

# ----------------- Action log + editable outcomes -----------------
st.subheader("📦 Action Log & Outcomes")
action_df = db.read_actions(username=username, limit=500)

if action_df.empty:
    st.info("No actions taken yet.")
else:
    st.caption("Edit the **outcome** column (success / failed / pending), then click Save.")
    editable = st.data_editor(
        action_df.rename(columns={"ts": "timestamp"}),
        num_rows="fixed",
        key="action_log_editor",
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "outcome": st.column_config.SelectboxColumn(
                "outcome",
                options=["pending", "success", "failed", "ignored"],
                required=True,
            ),
        },
        use_container_width=True,
    )

    if st.button("💾 Save Updated Outcomes"):
        original = action_df.set_index("id")["outcome"].to_dict()
        edited = editable.set_index("id")["outcome"].to_dict()
        changed = [(aid, new) for aid, new in edited.items() if original.get(aid) != new]
        for aid, new in changed:
            update_outcome(int(aid), str(new))
        st.success(f"Saved {len(changed)} outcome update(s).")
        st.rerun()

# ----------------- Outcome charts -----------------
st.subheader("📊 Outcome Analytics")
if not action_df.empty and "outcome" in action_df.columns:
    outcome_counts = action_df["outcome"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(outcome_counts, labels=outcome_counts.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    if "ts" in action_df.columns:
        action_df["date"] = pd.to_datetime(action_df["ts"], errors="coerce").dt.date
        actions_by_day = action_df.groupby(["date", "outcome"]).size().unstack(fill_value=0)
        if not actions_by_day.empty:
            st.bar_chart(actions_by_day)

# ----------------- Learning insights -----------------
st.subheader("🧠 Agent Learning Insights")
success_rates = get_action_success_rate(username=username)
if hasattr(success_rates, "empty") and not success_rates.empty:
    st.write("Action success rates (this user):")
    for action, rate in success_rates.items():
        st.write(f"• {action}: **{rate:.0%}**")
else:
    st.info("No success data yet — mark outcomes to start learning!")

# ----------------- Forecast -----------------
st.subheader("🔮 Revenue Forecast")
try:
    forecast_df = forecast_revenue(user_csv_path, periods=10)
    if not forecast_df.empty:
        st.line_chart(forecast_df.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
        future_anom = forecast_df[forecast_df["anomaly"]]
        if not future_anom.empty:
            st.warning(f"⚠️ Potential future anomalies: {len(future_anom)}")
            st.dataframe(future_anom[["ds", "yhat", "anomaly"]])
        else:
            st.success("✅ No significant anomalies forecasted.")
    else:
        st.info("Not enough history for a forecast yet.")
except Exception as e:
    st.info(f"Forecast not available ({e}).")
