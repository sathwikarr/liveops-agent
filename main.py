"""CLI / Streamlit entry that delegates to the canonical runner.

The actual pipeline lives in `pages/run_agent.py` so the dashboard's
"▶️ Run Agent Once" button and this file share the same code path.
"""
from __future__ import annotations

import streamlit as st

from pages.run_agent import run_liveops_agent


if __name__ == "__main__":
    if "username" in st.session_state:
        run_liveops_agent(st.session_state["username"])
    else:
        st.warning("⛔ Please log in to continue.")
        try:
            st.switch_page("app.py")
        except Exception:
            st.stop()
