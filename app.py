"""LiveOps Agent — public landing page.

This is the entry point — open it with `streamlit run app.py` and you land
HERE, not at the login. The point: someone clicking the GitHub link in a
README should be able to see what the product *does* without making an
account first.

- "Try the demo" → `pages/demo.py` (auto-loaded sample data, no auth)
- "Open Workbench" → `pages/analyst_workbench.py` (anonymous browsing OK)
- "Log in / Sign up" → `pages/login.py` (then on to ops dashboard)
"""
from __future__ import annotations

import streamlit as st
from dotenv import find_dotenv, load_dotenv

# Load .env early so any downstream agent module imports see env vars.
load_dotenv(find_dotenv(), override=False)

from agent import db, ui

st.set_page_config(
    page_title="LiveOps Agent — AI co-pilot for retail ops",
    page_icon="📈",
    layout="wide",
)

# Make sure tables exist on first ever boot (cheap, idempotent).
db.init_db()

# Shared sidebar chrome — nav links + login state. No anonymous-banner here
# because the page itself IS the CTA.
ui.apply_chrome("app.py", show_anon_banner=False)


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #

hero_left, hero_right = st.columns([3, 2], gap="large")

with hero_left:
    st.markdown("# 📈 **LiveOps Agent**")
    st.markdown(
        "### An AI co-pilot for retail / ecommerce ops."
    )
    st.write(
        "Watches your live order stream, catches anomalies the dashboard "
        "would miss, asks an LLM **why**, picks a mitigating action with a "
        "Thompson-sampling bandit, and routes severity-gated alerts to "
        "Slack + email. Ships with a 9-stage **Analyst Workbench** that "
        "turns a CSV into RFM segments, elasticity curves, churn predictions, "
        "and a recommendation calendar in two clicks."
    )

    cta1, cta2, cta3 = st.columns(3)
    with cta1:
        if st.button("🎬  Try the demo", type="primary", use_container_width=True):
            st.switch_page("pages/demo.py")
    with cta2:
        if st.button("🧪  Open Workbench", use_container_width=True):
            st.switch_page("pages/analyst_workbench.py")
    with cta3:
        if ui.is_authed():
            if st.button("📊  Ops Dashboard", use_container_width=True):
                st.switch_page("pages/dashboard.py")
        else:
            if st.button("🔐  Log in / Sign up", use_container_width=True):
                st.switch_page("pages/login.py")

    st.caption(
        "✨ The demo and workbench require **no signup**. Login is only "
        "needed to persist work + run the realtime ops loop."
    )

with hero_right:
    # Tiny stats block — the kind of "look how serious this is" panel that
    # turns a portfolio link into a conversation starter.
    st.markdown(" ")  # vertical breathing room
    st.markdown(
        """
        <div style='border:1px solid rgba(127,127,127,0.25);border-radius:12px;
                    padding:18px;background:rgba(127,127,127,0.05)'>
        <div style='font-size:13px;color:#888;letter-spacing:1px;text-transform:uppercase'>
        At a glance
        </div>
        <table style='width:100%;font-size:15px;border-collapse:collapse;margin-top:6px'>
          <tr><td style='padding:6px 0'>📦 Tests</td><td style='text-align:right'><b>226 passing</b></td></tr>
          <tr><td style='padding:6px 0'>🧩 Pipeline stages</td><td style='text-align:right'><b>9 (analyst) + 4 (ops)</b></td></tr>
          <tr><td style='padding:6px 0'>🔌 Live connectors</td><td style='text-align:right'><b>Postgres · Sheets · S3 · File</b></td></tr>
          <tr><td style='padding:6px 0'>🤖 LLM agent</td><td style='text-align:right'><b>10 tools, dual backend</b></td></tr>
          <tr><td style='padding:6px 0'>📊 Charts</td><td style='text-align:right'><b>8 Plotly types</b></td></tr>
          <tr><td style='padding:6px 0'>🐳 Deploy</td><td style='text-align:right'><b>Docker · Fly.io · GHCR</b></td></tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# --------------------------------------------------------------------------- #
# What you can do — feature strip
# --------------------------------------------------------------------------- #

st.markdown("## What you can do here")

f1, f2, f3 = st.columns(3, gap="medium")

with f1:
    st.markdown("### 🎬  See it on sample data")
    st.write(
        "The **Demo** page auto-loads a synthetic retail dataset — 3,878 "
        "orders × 30 SKUs × 200 customers — and walks you through every "
        "stage with interactive Plotly charts. Best place to start."
    )
    if st.button("Open the Demo →", key="card_demo", use_container_width=True):
        st.switch_page("pages/demo.py")

with f2:
    st.markdown("### 🧪  Run it on your data")
    st.write(
        "The **Analyst Workbench** lets you upload a CSV / XLSX / JSON / "
        "Parquet, or wire up a saved Postgres / Google Sheets / S3 "
        "connection. Then ingest → clean → analyze → predict → recommend."
    )
    if st.button("Open the Workbench →", key="card_wb", use_container_width=True):
        st.switch_page("pages/analyst_workbench.py")

with f3:
    st.markdown("### 📊  Run the ops loop")
    st.write(
        "The **Ops Dashboard** ingests your live order stream, runs anomaly "
        "detection + bandit-picked actions on a schedule, and routes alerts "
        "to Slack + email. Requires login so your work persists."
    )
    if ui.is_authed():
        if st.button("Open the Dashboard →", key="card_dash",
                     use_container_width=True):
            st.switch_page("pages/dashboard.py")
    else:
        if st.button("Log in / Sign up →", key="card_login",
                     use_container_width=True):
            st.switch_page("pages/login.py")


st.divider()


# --------------------------------------------------------------------------- #
# How it works — pipeline diagram (ASCII so it always renders)
# --------------------------------------------------------------------------- #

st.markdown("## How it works")

st.markdown(
    """
The two products share a SQLite memory, a Fernet-encrypted connection store,
and the bcrypt auth layer. Either path can stand alone.
"""
)

p1, p2 = st.columns(2, gap="large")

with p1:
    st.markdown("#### 🟣 Analyst pipeline")
    st.code(
        "ingest → eda → clean → analyze → predict → recommend → calendar\n"
        "                                       │\n"
        "                                       ├─ pinboard (HTML export)\n"
        "                                       └─ LLM agent (10 tools)",
        language="text",
    )

with p2:
    st.markdown("#### 🔴 Ops loop")
    st.code(
        "stream → detect (z-score + IForest) → explain (Gemini) →\n"
        "         bandit pick → simulate → notify (Slack + email) →\n"
        "         memory (SQLite)",
        language="text",
    )


st.divider()


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #

footer_left, footer_right = st.columns([3, 2])

with footer_left:
    st.caption(
        "Built by Sathwik Arroju · "
        "[GitHub](https://github.com/sathwikarr/liveops-agent) · "
        "MIT license · 226 tests · Python 3.11+"
    )

with footer_right:
    if ui.is_authed():
        st.caption(f"You're logged in as **{ui.current_username()}**.")
    else:
        st.caption(
            "👋 Browsing anonymously. "
            "[Log in or sign up](/login) to persist your work."
        )
