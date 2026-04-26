"""LiveOps Agent — Streamlit entry point.

Run with:

    streamlit run app.py

This is the login page. After login, Streamlit's multipage routing exposes
`pages/dashboard.py` and `pages/run_agent.py` automatically.

The previous insurance-premium demo that lived here has been preserved as
`app_insurance_demo.py` for reference.
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load .env early so agent/action.py and agent/explain.py see env vars
load_dotenv(find_dotenv(), override=False)

st.set_page_config(page_title="LiveOps Agent — Login", page_icon="🔐", layout="centered")


def login_page() -> None:
    st.title("🔐 LiveOps Agent")
    st.caption("Real-time anomaly detection + LLM-powered explanations.")

    if "username" in st.session_state:
        st.success(f"Already logged in as **{st.session_state['username']}**")
        if st.button("Go to Dashboard →"):
            st.switch_page("pages/dashboard.py")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()
        return

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        submitted = st.form_submit_button("Log In")

    if submitted:
        cleaned = (username or "").strip()
        # Basic input hygiene — block path traversal in user_data/{username}.csv
        if not cleaned:
            st.error("Please enter a username.")
        elif any(c in cleaned for c in r"/\:.") or cleaned in {".", ".."}:
            st.error("Username can't contain `/`, `\\`, `:`, or `.`")
        else:
            st.session_state["username"] = cleaned
            st.success(f"Welcome, {cleaned}! Redirecting…")
            st.switch_page("pages/dashboard.py")


if __name__ == "__main__":
    login_page()
else:
    # Streamlit imports the file with __name__ == "__main__" already, but
    # call this defensively so the page renders if the runtime ever changes.
    login_page()
