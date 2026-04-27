"""Demo route — auto-loads the bundled retail dataset and walks through
every analyst stage. This is the "land here, get convinced in 60 seconds"
page recruiters and stakeholders see first.

No upload, no auth — just hit the page and watch the pipeline run.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from analyst import (
    analysis as A,
    calendar as Cal,
    charts as Ch,
    clean as C,
    eda as E,
    ingest as I,
    predict as P,
    recommend as RC,
)


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "analyst" / "sample_data"
ORDERS_PATH = SAMPLE_DIR / "retail_orders.csv"
INVENTORY_PATH = SAMPLE_DIR / "inventory.csv"

# Role map for the bundled dataset — keys are the analyst's canonical
# role names; values are the actual column names in retail_orders.csv.
ROLE_MAP = {
    "customer": "customer_id",
    "date": "order_date",
    "amount": "revenue",
    "product": "product_id",
    "quantity": "qty",
    "price": "price",
    "region": "city",
}


st.set_page_config(page_title="Analyst Demo", page_icon="🎬", layout="wide")
st.title("🎬 Analyst Workbench — Live Demo")
st.caption(
    "A synthetic retail dataset is auto-loaded and run through every stage "
    "of the analyst pipeline. No upload required."
)

if not ORDERS_PATH.exists():
    st.error(f"Sample data missing at {ORDERS_PATH}. Run the data generator.")
    st.stop()


# --------------------------------------------------------------------------- #
# 1. Ingest
# --------------------------------------------------------------------------- #

st.subheader("1 · Ingest + schema inference")

with st.spinner("Loading sample dataset…"):
    ing = I.ingest(str(ORDERS_PATH))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(ing.df):,}")
c2.metric("Columns", len(ing.df.columns))
c3.metric("Detected kind", ing.kind.value if hasattr(ing.kind, "value") else str(ing.kind))
c4.metric("Memory", f"{ing.df.memory_usage(deep=True).sum() / 1024:.0f} KB")

with st.expander("Inferred schema"):
    sch = pd.DataFrame([
        {"column": c.name, "type": c.dtype, "nulls": c.null_count, "reason": c.reason}
        for c in ing.schema.columns
    ])
    st.dataframe(sch, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 2. EDA
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("2 · Exploratory data analysis")

with st.spinner("Profiling columns…"):
    eda = E.profile(ing.df, ing.schema)

c1, c2, c3 = st.columns(3)
c1.metric("Outlier rows flagged", len(eda.outliers))
c2.metric("Strong correlations (|r|≥0.5)",
          sum(1 for cp in eda.correlations if abs(cp.r) >= 0.5))
seas = eda.seasonality
c3.metric(
    "Seasonality detected",
    "Yes" if (seas and (seas.has_weekly or seas.has_monthly)) else "No",
    help=f"Peak weekday: {getattr(seas, 'peak_weekday', None) or '—'} · "
         f"Peak month: {getattr(seas, 'peak_month', None) or '—'}",
)

ch1, ch2 = st.columns(2)
with ch1:
    if eda.correlations:
        cols = sorted(
            {cp.col_a for cp in eda.correlations}
            | {cp.col_b for cp in eda.correlations}
        )
        mat = pd.DataFrame(1.0, index=cols, columns=cols)
        for cp in eda.correlations:
            mat.loc[cp.col_a, cp.col_b] = cp.r
            mat.loc[cp.col_b, cp.col_a] = cp.r
        st.plotly_chart(Ch.correlation_heatmap(mat), use_container_width=True)
    else:
        st.info("No notable correlations.")

with ch2:
    st.plotly_chart(
        Ch.revenue_trend(ing.df, "order_date", "revenue", freq="W"),
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# 3. Clean
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("3 · Cleaning plan")
plan = C.propose(ing.df, ing.schema)
cleaned, audit = C.apply(ing.df, plan)
st.caption(
    f"Applied {len(plan.steps)} cleaning step(s) — "
    f"{len(audit)} change(s) logged."
)
if plan.steps:
    st.dataframe(
        pd.DataFrame([
            {"step": s.id, "column": s.column or "—",
             "label": s.label, "rationale": s.rationale,
             "applied": s.applied}
            for s in plan.steps
        ]),
        use_container_width=True, hide_index=True,
    )
else:
    st.success("Dataset is clean — no actions needed.")


# --------------------------------------------------------------------------- #
# 4. Analyze
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("4 · Analysis")

t_rfm, t_matrix, t_basket, t_elast = st.tabs(
    ["RFM segments", "Product matrix", "Market basket", "Elasticity"]
)

with t_rfm:
    rfm = A.rfm(cleaned, role_map=ROLE_MAP)
    st.plotly_chart(Ch.rfm_scatter(rfm), use_container_width=True)
    st.dataframe(rfm.head(20), use_container_width=True, hide_index=True)

with t_matrix:
    pm = A.product_matrix(cleaned, role_map=ROLE_MAP)
    st.plotly_chart(Ch.product_matrix(pm), use_container_width=True)
    st.dataframe(pm.head(30), use_container_width=True, hide_index=True)

with t_basket:
    basket = A.market_basket(cleaned, role_map=ROLE_MAP, basket_key="order_id")
    if basket.empty:
        st.info("No basket pairs found at the configured thresholds.")
    else:
        st.dataframe(basket.head(15), use_container_width=True, hide_index=True)

with t_elast:
    el = A.elasticity(cleaned, role_map=ROLE_MAP)
    if el is None or el.empty:
        st.info("Elasticity could not be estimated for any product.")
    else:
        # Pick the most price-sensitive product to highlight
        top = el.iloc[0]
        st.metric(f"Most elastic product: {top['product']}",
                  f"e = {top['elasticity']:.2f}")
        st.metric("R²", f"{top['r2']:.2f}")
        # Plot the actual price/qty for that product
        sub = cleaned[cleaned[ROLE_MAP["product"]] == top["product"]]
        st.plotly_chart(
            Ch.elasticity_scatter(
                sub[ROLE_MAP["price"]], sub[ROLE_MAP["quantity"]],
                slope=top["elasticity"],
                intercept=0.0,  # not stored on the row, fit visually
                r2=top["r2"],
            ),
            use_container_width=True,
        )
        with st.expander("All products by elasticity"):
            st.dataframe(el, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 5. Predict
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("5 · Predictions")

c_left, c_right = st.columns(2)

with c_left:
    st.markdown("**Churn risk distribution**")
    churn = P.churn_scores(cleaned, role_map=ROLE_MAP, churn_window_days=60)
    st.plotly_chart(Ch.churn_distribution(churn), use_container_width=True)

stockout = pd.DataFrame()
with c_right:
    st.markdown("**Stockout horizon (top 12 at risk)**")
    if INVENTORY_PATH.exists():
        inv = pd.read_csv(INVENTORY_PATH)
        # Build a per-product daily-demand from cleaned orders
        d = cleaned.copy()
        d[ROLE_MAP["date"]] = pd.to_datetime(d[ROLE_MAP["date"]], errors="coerce")
        days_window = (d[ROLE_MAP["date"]].max() - d[ROLE_MAP["date"]].min()).days or 1
        demand_map = (
            d.groupby(ROLE_MAP["product"])[ROLE_MAP["quantity"]].sum()
            / days_window
        ).to_dict()
        sk = inv.copy()
        sk["daily_demand"] = sk["product_id"].map(lambda pid: demand_map.get(pid, 0.0))
        sk["days_to_stockout"] = sk.apply(
            lambda r: float("inf") if r["daily_demand"] == 0
                      else round(r["on_hand"] / r["daily_demand"], 1),
            axis=1,
        )
        sk["risk"] = sk["days_to_stockout"].apply(
            lambda d: "Critical" if d <= 7
                      else ("Warning" if d <= 21 else "OK")
        )
        sk = sk.rename(columns={"product_id": "product"})
        sk = sk.sort_values("days_to_stockout").head(12)
        stockout = sk[["product", "on_hand", "daily_demand",
                       "days_to_stockout", "risk"]].copy()
        st.dataframe(stockout, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 6. Recommend
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("6 · Recommendations")

recs = RC.generate(
    rfm_df=rfm,
    matrix_df=pm,
    basket_df=basket if not basket.empty else None,
    elasticity_df=el if (el is not None and not el.empty) else None,
    churn_df=churn if not churn.empty else None,
    stockout_df=stockout if not stockout.empty else None,
    role_map=ROLE_MAP,
)[:10]

if not recs:
    st.info("No recommendations generated.")
else:
    for i, r in enumerate(recs, 1):
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{i}. {r.action}**")
                st.caption(f"{r.evidence}  ·  *{r.category} · {r.audience}*")
            with c2:
                st.metric("Score", f"{r.score:.2f}")
                st.caption(f"{r.confidence} · {r.impact_estimate}")


# --------------------------------------------------------------------------- #
# 7. Calendar
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("7 · Action calendar (next 8 weeks)")
cal = Cal.build_calendar(recs, weeks=8, start=date.today())
if cal.empty:
    st.info("No calendar entries.")
else:
    st.plotly_chart(Ch.calendar_gantt(cal), use_container_width=True)
    with st.expander("Calendar as table"):
        st.dataframe(cal, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #

st.divider()
st.caption(
    "Want to run this on your own data? Open the **Analyst Workbench** "
    "page from the sidebar and upload a CSV/Excel/JSON file. "
    "All charts here are live Plotly — click + drag to explore."
)
