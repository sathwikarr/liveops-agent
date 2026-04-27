"""Eval harness viewer — runs the agent against the eval corpus on demand,
shows pass rate, per-tool breakdown, and a per-case detail panel.

The page is anonymous-accessible. Recruiters land here, hit "Run evals",
and see real numbers come back in ~5 seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent import ui as _ui
from analyst.evals import ALL_CASES, run_all
from analyst.evals.cases import load_cases


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = REPO_ROOT / "analyst" / "sample_data" / "retail_orders.csv"
BASELINE = REPO_ROOT / "tests" / "fixtures" / "eval_baseline.json"


st.set_page_config(page_title="Agent Evals", page_icon="🧪", layout="wide")
_ui.apply_chrome("pages/evals.py")

st.title("🧪 LLM agent eval harness")
st.caption(
    "55 questions × 10 tools — graded on tool match, args match, "
    "no-forbidden tools, and must-succeed execution. Baseline pinned in "
    "`tests/fixtures/eval_baseline.json`; CI fails on any >2pp regression."
)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### Eval settings")
    backend = st.radio(
        "Planner backend",
        options=["heuristic", "auto", "llm"],
        index=0,
        help=("`heuristic` = pure regex routing (offline). "
              "`auto` falls back to heuristic if no GEMINI_API_KEY. "
              "`llm` forces Gemini."),
    )
    all_tags = sorted({t for c in ALL_CASES for t in c.tags})
    tag_filter = st.multiselect("Tags (filter, OR)", options=all_tags)
    run_btn = st.button("▶  Run evals", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Baseline panel
# --------------------------------------------------------------------------- #

baseline_col, _ = st.columns([1, 2])
with baseline_col:
    if BASELINE.exists():
        with st.container(border=True):
            base = json.loads(BASELINE.read_text())
            st.markdown("**Pinned baseline**")
            st.metric("Baseline pass rate", f"{base['pass_rate']*100:.1f}%",
                      help=f"{base['n_passed']}/{base['n_cases']} cases on "
                           f"backend={base['backend']}")
    else:
        st.info("No baseline pinned yet. Run evals → save a baseline JSON.")


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

if run_btn or st.session_state.get("eval_report"):
    if run_btn:
        if not SAMPLE_CSV.exists():
            st.error(f"Sample data missing: {SAMPLE_CSV}")
            st.stop()
        df = pd.read_csv(SAMPLE_CSV)
        cases = load_cases(tags=tag_filter or None)

        progress = st.progress(0.0, text=f"Running 0 / {len(cases)}…")
        last_id = st.empty()

        def _on_progress(i, n, r):
            progress.progress(i / n, text=f"Running {i} / {n}…")
            last_id.caption(
                f"Last: **{r.case_id}** — "
                f"{'✅ pass' if r.passed else '❌ fail'}"
            )

        report = run_all(cases, df, backend=backend, on_progress=_on_progress)
        st.session_state["eval_report"] = report
        progress.empty()
        last_id.empty()
    else:
        report = st.session_state["eval_report"]

    # ------------------------------------------------------------------- #
    # Top-line metrics
    # ------------------------------------------------------------------- #
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pass rate", f"{report.pass_rate*100:.1f}%",
              f"{report.n_passed}/{report.n_cases} cases")
    m2.metric("Mean overall score", f"{report.mean_overall:.3f}")
    m3.metric("Tool-pick accuracy", f"{report.mean_tool_match:.3f}")
    m4.metric("Args accuracy", f"{report.mean_args_match:.3f}")

    # Compare to baseline
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        delta = report.pass_rate - float(base["pass_rate"])
        if delta < -0.02:
            st.error(f"REGRESSION — {delta*100:+.1f}pp below pinned baseline.")
        elif delta < 0:
            st.warning(f"Below baseline by {abs(delta)*100:.1f}pp (within tolerance).")
        else:
            st.success(f"At or above pinned baseline ({delta*100:+.1f}pp).")

    st.divider()

    # ------------------------------------------------------------------- #
    # Per-tool breakdown — bar chart + table
    # ------------------------------------------------------------------- #
    st.subheader("Per-tool breakdown")
    if report.per_tool:
        tool_df = (pd.DataFrame.from_dict(report.per_tool, orient="index")
                   .reset_index().rename(columns={"index": "tool"})
                   .sort_values("pass_rate"))
        bar = px.bar(
            tool_df, x="pass_rate", y="tool",
            orientation="h", color="pass_rate",
            color_continuous_scale=[(0, "#ef4444"), (0.5, "#f59e0b"), (1, "#10b981")],
            range_color=(0, 1),
            hover_data={"n": True, "mean_overall": ":.3f", "pass_rate": ":.2%"},
            labels={"pass_rate": "Pass rate", "tool": "Tool"},
        )
        bar.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_showscale=False)
        bar.update_xaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(bar, use_container_width=True)
    else:
        st.info("No per-tool stats — empty corpus?")

    # ------------------------------------------------------------------- #
    # Per-case detail
    # ------------------------------------------------------------------- #
    st.subheader("Per-case detail")
    case_df = pd.DataFrame([
        {
            "id": r.case_id,
            "passed": "✅" if r.passed else "❌",
            "question": r.question,
            "planned": ", ".join(r.planned_tools) or "—",
            "tool_match": r.tool_match,
            "args_match": r.args_match,
            "success": r.success_match,
            "overall": r.overall,
        }
        for r in report.cases
    ])
    st.dataframe(case_df, use_container_width=True, hide_index=True,
                 column_config={
                     "tool_match": st.column_config.ProgressColumn(
                         "tool", format="%.2f", min_value=0.0, max_value=1.0),
                     "args_match": st.column_config.ProgressColumn(
                         "args", format="%.2f", min_value=0.0, max_value=1.0),
                     "success": st.column_config.ProgressColumn(
                         "success", format="%.2f", min_value=0.0, max_value=1.0),
                     "overall": st.column_config.ProgressColumn(
                         "overall", format="%.2f", min_value=0.0, max_value=1.0),
                 })

    # ------------------------------------------------------------------- #
    # Failures call-out
    # ------------------------------------------------------------------- #
    fails = [r for r in report.cases if not r.passed]
    if fails:
        st.subheader(f"Failures ({len(fails)})")
        for r in fails:
            with st.expander(f"❌ {r.case_id} — {r.question}"):
                st.markdown(f"**Planned tools:** `{r.planned_tools}`")
                st.markdown(f"**Reasons:**")
                for reason in r.failed_reasons:
                    st.markdown(f"- {reason}")
                if r.answer:
                    st.markdown("**Agent answer:**")
                    st.code(r.answer)
    else:
        st.success("🎉  No failures.")

    # ------------------------------------------------------------------- #
    # Download / pin
    # ------------------------------------------------------------------- #
    st.divider()
    json_blob = json.dumps(report.to_dict(), default=str, indent=2)
    st.download_button("Download report (JSON)", data=json_blob,
                       file_name=f"eval_report_{backend}.json",
                       mime="application/json")

else:
    st.info("Configure the backend + filters in the sidebar, then click "
            "**▶  Run evals**.")
