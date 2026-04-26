"""LiveOps agent runner.

Imported by `pages/dashboard.py` so the "▶️ Run Agent Once" button can fire
the full detect → explain → log → act pipeline against a user's CSV. Also
imported by `auto_agent.py` for the standalone autonomous loop.

Lives in `pages/` so Streamlit's multipage discovery picks it up alongside
the dashboard, and so the dashboard's existing
`from pages.run_agent import run_liveops_agent` import resolves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from agent.action import simulate_action
from agent.detect import detect_anomalies, read_latest_data
from agent.explain import explain_anomaly
from agent.memory import save_anomaly_log

REQUIRED_COLS = {"region", "product_id", "orders", "inventory", "revenue"}


def _user_csv_path(username: str) -> Path:
    return REPO_ROOT / "user_data" / f"{username}.csv"


def run_pipeline(
    username: str,
    *,
    threshold: float = 1.5,
    top_k: int = 50,
    log: Callable[[str], None] = print,
    on_anomaly: Optional[Callable[[dict, str, str], None]] = None,
) -> int:
    """Run one pass of the LiveOps pipeline. Returns number of anomalies acted on.

    UI-agnostic: callers (Streamlit page, auto_agent.py CLI) pass `log` and
    `on_anomaly` callbacks. `on_anomaly(row_dict, explanation, action)` runs
    after each anomaly is processed.
    """
    csv_path = _user_csv_path(username)
    if not csv_path.exists():
        log(f"⚠️ No CSV for user {username!r} at {csv_path}")
        return 0

    df = read_latest_data(str(csv_path), n=2000)
    if df is None or df.empty:
        log("ℹ️ No rows to analyze yet.")
        return 0

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        log(f"❌ CSV is missing required columns: {', '.join(sorted(missing))}")
        return 0

    anomalies = detect_anomalies(df, threshold=threshold).head(top_k)
    if anomalies.empty:
        log("✅ No anomalies found.")
        return 0

    count = 0
    for _, row in anomalies.iterrows():
        region = row["region"]
        product_id = row["product_id"]
        metric = row["metric"]
        z = float(row["z_score"])
        revenue = float(row["revenue"])
        orders = int(row["orders"])
        inventory = int(row["inventory"])

        explanation = explain_anomaly(region, product_id, orders, inventory, revenue)

        try:
            save_anomaly_log(
                region, product_id, orders, inventory, revenue, explanation,
                username=username, metric=metric, z_score=z,
            )
        except Exception as e:
            log(f"⚠️ Could not save anomaly: {e}")

        action = "ℹ️ Log only – no action needed"
        try:
            action = simulate_action(username, region, product_id, orders, inventory, revenue)
        except Exception as e:
            log(f"⚠️ Action layer failed: {e}")

        if on_anomaly:
            on_anomaly(dict(row), explanation, action)

        count += 1

    log(f"📦 Processed {count} anomalies for {username}.")
    return count


# --------------------------------------------------------------------------- #
# Streamlit page rendering
# --------------------------------------------------------------------------- #

def run_liveops_agent(username: str) -> None:
    """Streamlit-flavored wrapper used by the dashboard's 'Run Agent Once' button."""
    import streamlit as st

    st.write("🧠 LiveOps Agent Running…")

    msgs: list[str] = []

    def _log(m: str) -> None:
        msgs.append(m)

    def _on_anomaly(row: dict, explanation: str, action: str) -> None:
        title = (
            f"🚨 {row['product_id']} in {row['region']} "
            f"— {row['metric']}={row['value']:.2f} (z={row['z_score']:+.2f}, {row['scope']})"
        )
        with st.expander(title, expanded=False):
            st.markdown(explanation)
            st.write("📦 Action:", action)

    n = run_pipeline(username, log=_log, on_anomaly=_on_anomaly)
    for m in msgs:
        st.write(m)
    if n == 0 and not any("anomalies" in m for m in msgs):
        st.success("✅ Done — no anomalies found.")


# Render a stub when Streamlit auto-loads this as a page
def _is_streamlit_runtime() -> bool:
    try:
        import streamlit.runtime.scriptrunner as sr
        return sr.get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit_runtime():
    import streamlit as st
    st.set_page_config(page_title="Run Agent")
    st.title("▶️ Run LiveOps Agent")
    if "username" not in st.session_state:
        st.warning("⛔ Please log in first.")
    else:
        run_liveops_agent(st.session_state["username"])
