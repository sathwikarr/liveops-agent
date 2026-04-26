"""LiveOps Agent — Streamlit entry point.

Run with:

    streamlit run app.py

Login + signup, both backed by SQLite (`agent.db.users`) with bcrypt-hashed
passwords. After login, Streamlit's multipage routing exposes
`pages/dashboard.py` and `pages/run_agent.py`.

The previous insurance-premium demo that lived here is preserved as
`app_insurance_demo.py`.
"""
from __future__ import annotations

import streamlit as st
from dotenv import find_dotenv, load_dotenv

# Load .env early so agent/action.py and agent/explain.py see env vars
load_dotenv(find_dotenv(), override=False)

from agent import auth, db

st.set_page_config(page_title="LiveOps Agent — Login", page_icon="🔐", layout="centered")


def _render_logged_in() -> None:
    st.success(f"Logged in as **{st.session_state['username']}**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Dashboard →", use_container_width=True):
            st.switch_page("pages/dashboard.py")
    with col2:
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def _render_login_tab() -> None:
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In", use_container_width=True)

    if not submitted:
        return

    ok, msg = auth.login((username or "").strip(), password or "")
    if not ok:
        st.error(msg)
        return
    st.session_state["username"] = username.strip()
    st.success(msg)
    st.switch_page("pages/dashboard.py")


def _render_signup_tab() -> None:
    with st.form("signup_form", clear_on_submit=False):
        username = st.text_input("Choose a username", key="signup_username",
                                 help="3–32 chars: letters, digits, `_`, `-`")
        password = st.text_input("Choose a password", type="password",
                                 key="signup_password", help="At least 8 characters")
        confirm = st.text_input("Confirm password", type="password",
                                key="signup_confirm")
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if not submitted:
        return

    if password != confirm:
        st.error("Passwords don't match.")
        return

    ok, msg = auth.signup((username or "").strip(), password or "")
    if not ok:
        st.error(msg)
        return
    st.success(msg + " Switch to the Log In tab.")


def main() -> None:
    # Make sure tables exist on first ever boot
    db.init_db()

    st.title("🔐 LiveOps Agent")
    st.caption("Real-time anomaly detection + LLM-powered explanations.")

    if "username" in st.session_state:
        _render_logged_in()
        return

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])
    with login_tab:
        _render_login_tab()
    with signup_tab:
        _render_signup_tab()


main()
