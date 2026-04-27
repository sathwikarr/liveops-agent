"""Analyst Workbench — schema-agnostic data-science page.

Drop any CSV/Excel/JSON in the uploader and walk through:
    1. Ingest      — what does this dataset look like?
    2. EDA         — distributions, correlations, outliers, seasonality
    3. Clean       — proposed fixes with audit log (toggleable)
    4. Analysis    — RFM, cohorts, basket, elasticity, product matrix
    5. Predict     — churn, stockout horizon
    6. Recommend   — business actions sorted by score
    7. Ask / What-if / Report
    8. Action calendar + multi-CSV join + competitor stub

The whole thing is a single page so users can keep state via st.session_state
without juggling tabs.
"""
from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from analyst import (
    analysis as A, calendar as Cal, clean as C, competitor as Comp,
    eda as E, ingest as I, join as J, nlq as N, predict as P,
    recommend as RC, report as R, whatif as W,
)

st.set_page_config(page_title="Analyst Workbench", layout="wide")
st.title("🧠 Analyst Workbench")
st.caption("Drop any tabular dataset — the workbench does ingestion, EDA, cleaning, analysis, prediction, and recommendations.")

# Soft auth: if the user logged into the LiveOps app, reuse their username for bandit feedback.
username = st.session_state.get("username")
if not username:
    st.info("Tip: log in via the main app to get personalised bandit-weighted recommendations.")

# --------------------------------------------------------------------------- #
# 1. Ingest
# --------------------------------------------------------------------------- #
st.header("1. Ingest")

# Saved-connection store lives next to the rest of the app's user data
from analyst.connectors import (
    ConnectionStore, SavedConnection, REGISTRY as CONN_REG,
    ConnectionError as ConnErr,
)
_conn_db = REPO_ROOT / "user_data" / "connections.db"
_conn_store = ConnectionStore(_conn_db)

src_tabs = st.tabs(["📁 File upload", "💾 Saved connection", "➕ New connection"])

if "ingest" not in st.session_state:
    st.session_state.ingest = None

with src_tabs[0]:
    uploaded = st.file_uploader(
        "Upload CSV / Excel / JSON / JSONL",
        type=["csv", "tsv", "xlsx", "xls", "json", "jsonl"],
        key="upl_main",
    )
    sample = st.checkbox("Use bundled sample (joe.csv) instead", value=False)
    if uploaded is not None:
        raw = uploaded.read()
        st.session_state.ingest = I.ingest(raw)
    elif sample:
        sample_path = REPO_ROOT / "user_data" / "joe.csv"
        if sample_path.exists():
            st.session_state.ingest = I.ingest(sample_path)
        else:
            st.warning("Sample file not found.")

with src_tabs[1]:
    saved = _conn_store.list()
    if not saved:
        st.info("No saved connections yet — create one in the next tab.")
    else:
        names = [c.name for c in saved]
        kinds = {c.name: c.kind for c in saved}
        pick = st.selectbox(
            "Connection",
            names,
            format_func=lambda n: f"{n}  ·  {kinds[n]}",
        )
        cc1, cc2 = st.columns([1, 1])
        if cc1.button("🔄 Fetch / refresh", type="primary", key="conn_run"):
            chosen = _conn_store.get(pick)
            try:
                with st.spinner(f"Fetching from {chosen.kind}…"):
                    res = chosen.connector().fetch()
                # Re-run ingest's schema inference on the fetched DataFrame
                st.session_state.ingest = I.ingest(res.df)
                st.success(f"Loaded {res.rows:,} rows from {res.source}")
            except ConnErr as e:
                st.error(f"Connection failed: {e}")
        if cc2.button("🗑️ Delete", key="conn_del"):
            _conn_store.delete(pick)
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

with src_tabs[2]:
    kind = st.selectbox("Type", list(CONN_REG.keys()), key="new_kind")
    name = st.text_input("Connection name (must be unique)", key="new_name")
    schema = CONN_REG[kind].param_schema
    new_params: dict = {}
    for pname, ptype in schema.items():
        if ptype == "secret":
            new_params[pname] = st.text_input(
                pname, type="password", key=f"new_{pname}"
            )
        else:
            new_params[pname] = st.text_input(pname, key=f"new_{pname}")
    if st.button("💾 Save connection", key="new_save"):
        try:
            sc = SavedConnection(
                name=name.strip(), kind=kind,
                params={k: v for k, v in new_params.items() if v},
            )
            _conn_store.save(sc)
            st.success(f"Saved '{sc.name}' — switch to the Saved tab to fetch.")
        except Exception as e:
            st.error(f"Could not save: {e}")

result = st.session_state.ingest
if result is None:
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{result.n_rows:,}")
c2.metric("Columns", result.n_cols)
c3.metric("Detected kind", f"{result.kind.kind} ({result.kind.confidence:.0%})")

with st.expander("Schema (auto-inferred)"):
    schema_rows = [{
        "column": c.name, "type": c.inferred_type,
        "null %": f"{c.null_pct:.1%}", "unique %": f"{c.unique_pct:.1%}",
        "reason": c.reason,
    } for c in result.schema.columns]
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

if result.issues:
    with st.expander(f"Data issues ({len(result.issues)})", expanded=True):
        for i in result.issues:
            st.warning(i)

with st.expander("Role map (column → semantic role)"):
    st.json(result.kind.role_map)

# --------------------------------------------------------------------------- #
# 2. EDA
# --------------------------------------------------------------------------- #
st.header("2. Exploratory Data Analysis")

eda_report = E.profile(result.df, result.schema, kind=result.kind.kind, role_map=result.kind.role_map)
st.session_state.eda = eda_report

st.subheader("Headline")
for line in eda_report.headline:
    st.write("• " + line)

with st.expander("Plain-English summary (Gemini if configured)"):
    if st.button("Generate narrative", key="gen_nar"):
        with st.spinner("Writing summary…"):
            st.session_state.eda_narrative = E.narrate(eda_report, kind=result.kind.kind)
    if st.session_state.get("eda_narrative"):
        st.write(st.session_state.eda_narrative)

if eda_report.numeric:
    st.subheader("Distributions")
    cols = st.columns(3)
    for i, n in enumerate(eda_report.numeric[:6]):
        ax_col = cols[i % 3]
        with ax_col:
            fig, ax = plt.subplots(figsize=(4, 2.2))
            ax.bar(range(len(n.histogram)), n.histogram, color="#3b82f6")
            ax.set_title(n.name, fontsize=10)
            ax.set_xticks([])
            st.pyplot(fig, clear_figure=True)

if eda_report.correlations:
    st.subheader("Top correlations")
    st.dataframe(
        pd.DataFrame([c.to_dict() for c in eda_report.correlations]),
        use_container_width=True, hide_index=True,
    )

if eda_report.outliers:
    st.subheader("Outliers")
    st.dataframe(
        pd.DataFrame([o.to_dict() for o in eda_report.outliers]),
        use_container_width=True, hide_index=True,
    )

if eda_report.seasonality and eda_report.seasonality.notes:
    st.subheader("Seasonality")
    for n in eda_report.seasonality.notes:
        st.info(n)

# --------------------------------------------------------------------------- #
# 3. Clean
# --------------------------------------------------------------------------- #
st.header("3. Cleaning")

plan = C.propose(result.df, result.schema, kind=result.kind.kind, role_map=result.kind.role_map)
st.write(f"Proposed {len(plan.steps)} cleaning step(s). Untick any you want to skip.")

for s in plan.steps:
    s.applied = st.checkbox(f"{s.label} — *{s.rationale}*", value=s.applied, key=f"step_{s.id}")

if st.button("Apply cleaning"):
    cleaned, audit = C.apply(result.df, plan)
    st.session_state.cleaned = cleaned
    st.session_state.audit = audit

cleaned = st.session_state.get("cleaned", result.df)
audit = st.session_state.get("audit", [])
if audit:
    st.success(f"Applied {len(audit)} cleaning step(s).")
    st.dataframe(pd.DataFrame([a.to_dict() for a in audit]), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# 4. Analysis
# --------------------------------------------------------------------------- #
st.header("4. Analysis")

rmap = result.kind.role_map
rfm_df = A.rfm(cleaned, rmap)
matrix_df = A.product_matrix(cleaned, rmap)
basket_df = A.market_basket(cleaned, rmap)
elast_df = A.elasticity(cleaned, rmap)
trend_df = A.revenue_trend(cleaned, rmap, freq="W")

t1, t2, t3, t4, t5 = st.tabs(["RFM", "Product matrix", "Basket", "Elasticity", "Revenue trend"])
with t1:
    if rfm_df.empty: st.info("Need customer + date + amount roles for RFM.")
    else:
        st.dataframe(rfm_df.head(50), use_container_width=True, hide_index=True)
        st.bar_chart(rfm_df["segment"].value_counts())
with t2:
    if matrix_df.empty: st.info("Need product + amount + date roles.")
    else: st.dataframe(matrix_df, use_container_width=True, hide_index=True)
with t3:
    if basket_df.empty: st.info("Need product role and a basket key (or customer+date).")
    else: st.dataframe(basket_df, use_container_width=True, hide_index=True)
with t4:
    if elast_df.empty: st.info("Need product + quantity (and price or amount).")
    else: st.dataframe(elast_df, use_container_width=True, hide_index=True)
with t5:
    if trend_df.empty: st.info("Need date + amount roles.")
    else:
        st.dataframe(trend_df.tail(20), use_container_width=True, hide_index=True)
        st.line_chart(trend_df.set_index("period")[["revenue", "rolling_4"]])

# --------------------------------------------------------------------------- #
# 5. Predict
# --------------------------------------------------------------------------- #
st.header("5. Prediction")

churn_df = P.churn_scores(cleaned, rmap)
stock_df = P.stockout_horizon(cleaned, rmap)

pc1, pc2 = st.columns(2)
with pc1:
    st.subheader("Churn risk")
    if churn_df.empty: st.info("Need customer + date for churn.")
    else:
        st.dataframe(churn_df.head(50), use_container_width=True, hide_index=True)
        st.bar_chart(churn_df["risk"].value_counts())
with pc2:
    st.subheader("Stockout horizon")
    if stock_df.empty: st.info("Need product + date + quantity (+ optional inventory).")
    else: st.dataframe(stock_df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# 6. Recommendations
# --------------------------------------------------------------------------- #
st.header("6. Recommendations")

recs = RC.generate(
    rfm_df=rfm_df, elasticity_df=elast_df, matrix_df=matrix_df,
    basket_df=basket_df, churn_df=churn_df, stockout_df=stock_df,
    seasonality=eda_report.seasonality, role_map=rmap,
    bandit_username=username,
)
st.session_state.recs = recs

if not recs:
    st.info("No recommendations — once you have customers/products/forecasts, this fills up.")
else:
    for r in recs[:25]:
        with st.expander(f"[{r.confidence}] {r.action}"):
            st.write(f"**Why:** {r.evidence}")
            st.write(f"**Impact:** {r.impact_estimate}")
            st.write(f"**Audience:** {r.audience}  ·  **Category:** {r.category}")
            if username:
                col_a, col_b = st.columns(2)
                if col_a.button("✅ Did this", key=f"good_{r.bandit_arm}"):
                    try:
                        from agent.db import insert_action
                        insert_action(username=username, region="-", product_id="-",
                                      action=r.bandit_arm, outcome="success")
                        st.success("Logged success — bandit will weight this arm higher next time.")
                    except Exception as e:
                        st.warning(f"Could not log: {e}")
                if col_b.button("❌ Didn't work", key=f"bad_{r.bandit_arm}"):
                    try:
                        from agent.db import insert_action
                        insert_action(username=username, region="-", product_id="-",
                                      action=r.bandit_arm, outcome="failed")
                        st.success("Logged failure.")
                    except Exception as e:
                        st.warning(f"Could not log: {e}")

# --------------------------------------------------------------------------- #
# 7. Ask / What-if / Report
# --------------------------------------------------------------------------- #
st.header("7. Ask · What-if · Report")

a, agent_tab, b, c = st.tabs(
    ["Natural-language query", "🤖 Agent (tool-using)", "What-if simulator", "Narrative report"]
)

with a:
    q = st.text_input("Ask anything", value="What's the total revenue?")
    if st.button("Ask", key="nlq_btn"):
        with st.spinner("Thinking…"):
            ans = N.ask(q, cleaned, rmap)
        st.write(ans.answer)
        if ans.expression:
            st.code(ans.expression, language="python")
        if ans.data is not None:
            st.dataframe(ans.data, use_container_width=True, hide_index=True)

with agent_tab:
    st.markdown(
        "Ask a high-level question and the agent picks which analyst tools "
        "to run, executes them on the live data, then synthesizes a "
        "natural-language answer. Works offline via keyword routing — "
        "uses Gemini when `GEMINI_API_KEY` is set."
    )
    from analyst import agent as Agent

    examples = [
        "Which customers are most likely to churn?",
        "What are the top 5 products by revenue?",
        "Show me the price elasticity per product",
        "RFM segments breakdown",
        "Co-purchase pairs",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, key=f"ex_{ex[:10]}"):
            st.session_state["agent_q"] = ex

    aq = st.text_input(
        "Question",
        value=st.session_state.get("agent_q", "Which customers are most likely to churn?"),
        key="agent_q_input",
    )
    backend = st.radio(
        "Backend", ["auto", "heuristic", "llm"], horizontal=True,
        help="auto = LLM if API key present, else heuristic. heuristic = no API call.",
    )
    if st.button("Run agent", type="primary", key="agent_run"):
        with st.spinner("Planning + running tools…"):
            res = Agent.ask(aq, cleaned, rmap, backend=backend)
        st.success(res.answer)
        with st.expander(f"Plan ({res.plan.backend} backend, {len(res.plan.steps)} step(s))"):
            for i, step in enumerate(res.plan.steps, 1):
                st.markdown(f"**{i}. `{step.tool}`** — {step.why}")
                if step.args:
                    st.json(step.args)
        with st.expander("Tool observations"):
            for obs in res.observations:
                st.markdown(f"**`{obs['tool']}`**")
                st.json(obs["result"])

with b:
    if elast_df.empty:
        st.info("Need an elasticity table to simulate price changes.")
    else:
        prods = elast_df["product"].astype(str).tolist()
        sel = st.selectbox("Product", prods)
        pct = st.slider("Price change %", -30, 30, -10)
        if sel:
            row = elast_df[elast_df["product"] == sel].iloc[0].to_dict()
            res_w = W.simulate_price_change(row, pct_change=pct)
            st.json(res_w)

with c:
    if R.HAS_DOCX:
        if st.button("Build .docx report"):
            out_dir = REPO_ROOT / "user_data" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"analyst_report_{date.today()}.docx"
            narrative = st.session_state.get("eda_narrative") or ""
            R.build_report(
                out_path,
                ingest_result=result, eda_report=eda_report, narrative=narrative,
                rfm_df=rfm_df, matrix_df=matrix_df, elasticity_df=elast_df,
                churn_df=churn_df, stockout_df=stock_df, basket_df=basket_df,
                recommendations=recs,
            )
            st.success(f"Saved to {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download report", f.read(), file_name=out_path.name)
    else:
        st.warning("python-docx is not installed. Run: pip install python-docx")

# --------------------------------------------------------------------------- #
# 8. Calendar · Join · Competitor
# --------------------------------------------------------------------------- #
st.header("8. Calendar · Join · Competitor")

ta, tb, tc = st.tabs(["Action calendar", "Multi-CSV join", "Competitor"])

with ta:
    if not recs:
        st.info("Generate recommendations first.")
    else:
        weeks = st.slider("Weeks to plan", 2, 12, 6)
        cal_df = Cal.build_calendar(recs, weeks=weeks)
        st.dataframe(cal_df, use_container_width=True, hide_index=True)

with tb:
    second = st.file_uploader("Second dataset to join with the cleaned data", key="join_upload",
                              type=["csv", "tsv", "xlsx", "xls", "json", "jsonl"])
    if second is not None:
        right = I.read_any(second.read())
        sugg = J.suggest_keys(cleaned, right)
        st.write("Suggested join keys (top 5):")
        st.dataframe(pd.DataFrame([s.to_dict() for s in sugg]),
                     use_container_width=True, hide_index=True)
        joined, key = J.auto_join(cleaned, right)
        if key:
            st.success(f"Joined on {key.left_col} ↔ {key.right_col} (score {key.score:.2f}).")
            st.dataframe(joined.head(50), use_container_width=True, hide_index=True)
        else:
            st.warning("No high-confidence join key found.")

with tc:
    prods = []
    if "product" in rmap and rmap["product"] in cleaned.columns:
        prods = cleaned[rmap["product"]].dropna().astype(str).unique().tolist()[:10]
    if not prods:
        st.info("No product column to look up.")
    else:
        comp_df = Comp.lookup(prods, online=False)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("Plug a search MCP into Comp.lookup(..., online=True, fetch_fn=...) to enable real lookups.")
