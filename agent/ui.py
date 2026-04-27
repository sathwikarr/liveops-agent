"""Shared Streamlit chrome — sidebar nav, login state, and the
"sign up to save your work" banner shown on anonymous pages.

Every page calls `apply_chrome(page_name)` once at the top so the look + feel
stays consistent without repeating boilerplate.
"""
from __future__ import annotations

import streamlit as st


# Pages that work without authentication. Listed here so the sidebar can mark
# them as "✨ public" and the nav can hide auth-only pages from anonymous users.
PUBLIC_PAGES = {"app.py", "pages/demo.py", "pages/analyst_workbench.py",
                "pages/login.py", "pages/evals.py"}


def is_authed() -> bool:
    return bool(st.session_state.get("username"))


def current_username() -> str | None:
    return st.session_state.get("username")


def _sidebar_account_block() -> None:
    """Sidebar account widget: shows logout if authed, login CTA if not."""
    with st.sidebar:
        st.markdown("### Account")
        if is_authed():
            st.success(f"👤 {current_username()}")
            if st.button("🚪 Log out", use_container_width=True, key="ui_logout"):
                st.session_state.clear()
                st.switch_page("app.py")
        else:
            st.caption("You're browsing anonymously.")
            if st.button("🔐 Log in / Sign up", use_container_width=True,
                         key="ui_login_cta"):
                st.switch_page("pages/login.py")


def _sidebar_nav_block() -> None:
    """Sidebar nav links — explicit page links so users don't need to know
    that public pages are accessible while anonymous."""
    with st.sidebar:
        st.markdown("### Pages")
        st.page_link("app.py", label="🏠 Home", icon=None)
        st.page_link("pages/demo.py", label="🎬 Demo (no signup)")
        st.page_link("pages/analyst_workbench.py", label="🧪 Analyst Workbench")
        st.page_link("pages/evals.py", label="🧪 Agent Evals")
        if is_authed():
            st.page_link("pages/dashboard.py", label="📊 Ops Dashboard")
            st.page_link("pages/run_agent.py", label="🤖 Run Ops Agent")


def _anon_save_banner() -> None:
    """Shown at the top of pages that work anonymously — invites the user
    to sign up so their work persists across sessions."""
    if is_authed():
        return
    st.info(
        "👋 You're browsing anonymously — "
        "[**sign up**](#) to save connections, recommendations, and bandit feedback. "
        "Use the sidebar to log in.",
        icon="ℹ️",
    )


def apply_chrome(page_path: str, *, show_anon_banner: bool = True) -> None:
    """Idempotent chrome injector. Call once at the top of every page.

    page_path: this page's path relative to repo root (e.g. "pages/demo.py").
              Used to decide whether to show the anonymous banner.
    show_anon_banner: explicit override — set False on the landing page itself
                     so we don't double up on the CTA.
    """
    _sidebar_account_block()
    _sidebar_nav_block()
    if show_anon_banner and page_path in PUBLIC_PAGES and not is_authed():
        _anon_save_banner()


def require_auth(*, redirect_to: str = "pages/login.py") -> str:
    """Block until the user is authenticated. Returns the username on success;
    redirects to the login page otherwise."""
    if not is_authed():
        st.warning("You need to log in to view this page.")
        if st.button("Go to login", type="primary"):
            st.switch_page(redirect_to)
        st.stop()
    return current_username()  # type: ignore[return-value]


__all__ = [
    "apply_chrome", "require_auth", "is_authed", "current_username",
    "PUBLIC_PAGES",
]
