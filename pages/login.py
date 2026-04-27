"""Log in / sign up — separated from app.py so the landing page can stay public.

Backed by SQLite (`agent.db.users`) with bcrypt-hashed passwords. After login,
the user is bounced to the Ops Dashboard.
"""
from __future__ import annotations

import streamlit as st
from dotenv import find_dotenv, load_dotenv

# Load .env early so downstream agent modules see env vars.
load_dotenv(find_dotenv(), override=False)

from agent import auth, db, ui

st.set_page_config(page_title="LiveOps — Log In", page_icon="🔐", layout="centered")
ui.apply_chrome("pages/login.py", show_anon_banner=False)

db.init_db()


# --------------------------------------------------------------------------- #
# Already-logged-in branch
# --------------------------------------------------------------------------- #

if ui.is_authed():
    st.success(f"Logged in as **{ui.current_username()}**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Go to Dashboard →", use_container_width=True, type="primary"):
            st.switch_page("pages/dashboard.py")
    with c2:
        if st.button("Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    st.stop()


# --------------------------------------------------------------------------- #
# Anonymous branch — login + signup tabs
# --------------------------------------------------------------------------- #

st.title("🔐 LiveOps Agent")
st.caption("Log in to access the Ops Dashboard and persist your work. "
           "Or, [try the demo →](/demo) without signing up.")


def _render_login_tab() -> None:
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In", use_container_width=True,
                                          type="primary")

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
                                 key="signup_password",
                                 help="At least 8 characters")
        confirm = st.text_input("Confirm password", type="password",
                                key="signup_confirm")
        submitted = st.form_submit_button("Create account",
                                          use_container_width=True, type="primary")

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


login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])
with login_tab:
    _render_login_tab()
with signup_tab:
    _render_signup_tab()
