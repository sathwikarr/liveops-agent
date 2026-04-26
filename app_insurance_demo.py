import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

st.set_page_config(page_title="🏠 Insurance Premium Prediction", layout="wide")

STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","District of Columbia",
    "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine",
    "Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon",
    "Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia",
    "Washington","West Virginia","Wisconsin","Wyoming","Puerto Rico"
]

HIGH   = 1.30
MEDIUM = 1.10
LOW    = 1.00

HIGH_STATES = {"California","Florida","Louisiana","Texas","South Carolina","North Carolina","Alabama","Mississippi","Oklahoma","Hawaii","Puerto Rico"}
MEDIUM_STATES = {"Georgia","Virginia","Washington","Oregon","Arizona","New Mexico","Nevada","Colorado","New Jersey","New York","Massachusetts","Connecticut","Delaware","Maryland","District of Columbia","Pennsylvania","Illinois","Indiana","Ohio","Michigan","Missouri","Arkansas","Tennessee","Kentucky","Kansas","Nebraska"}
STATE_FACTOR = {s: (HIGH if s in HIGH_STATES else MEDIUM if s in MEDIUM_STATES else LOW) for s in STATES}

PROPERTY_TYPES = ["Single Family","Townhouse","Condo","Duplex","Multi-Family (2–4)","Manufactured","Mobile Home"]
TYPE_FACTOR = {"Single Family": 1.05, "Townhouse": 1.03, "Condo": 0.98, "Duplex": 1.01, "Multi-Family (2–4)": 1.08, "Manufactured": 1.06, "Mobile Home": 1.02}

def compute_quote(state, ptype, sqft, roof_age):
    base = 900
    hazard = STATE_FACTOR.get(state, LOW)
    prop = TYPE_FACTOR.get(ptype, 1.0)
    roof = 1.12 if roof_age > 15 else 1.06 if roof_age > 8 else 1.0
    size = min(1.40, 0.85 + (sqft or 0)/4000.0)
    premium = round(base * hazard * prop * roof * size)
    risk_lbl = "High" if premium > 1800 else "Medium" if premium > 1300 else "Low"
    return premium, risk_lbl

st.sidebar.title("Instant Quote")
address = st.sidebar.text_input("Street Address", "742 Evergreen Terrace")
state = st.sidebar.selectbox("State", STATES, index=STATES.index("Texas"))
ptype = st.sidebar.selectbox("Property Type", PROPERTY_TYPES, index=0)
sqft = st.sidebar.number_input("Square Footage", 300, 10000, 2200, 50)
roof_age = st.sidebar.number_input("Roof Age (years)", 0, 80, 8, 1)

st.title("🏠 Insurance Premium Prediction")
st.caption("Basic pricing by state risk tiers.")

premium, risk_lbl = compute_quote(state, ptype, sqft, roof_age)
c1, c2 = st.columns(2)
with c1:
    st.metric("Estimated Annual Premium", f"${premium:,}")
with c2:
    st.metric("Risk Classification", risk_lbl)

df_risk = pd.DataFrame({"state": STATES, "factor": [STATE_FACTOR[s] for s in STATES]})
top_risk = df_risk.sort_values("factor", ascending=False).head(12)

colA, colB = st.columns(2)
with colA:
    st.subheader("Highest State Risk Factors (Top 12)")
    st.altair_chart(
        alt.Chart(top_risk).mark_bar().encode(
            x=alt.X("state:N", sort="-y"),
            y="factor:Q"
        ),
        use_container_width=True
    )

trend = pd.DataFrame({
    "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "premium": [1170,1180,1200,1210,1245,1230,1265,1280,1305,1310,1320,1335]
})
with colB:
    st.subheader("Average Premium Trend (Demo)")
    st.altair_chart(
        alt.Chart(trend).mark_line(point=True).encode(
            x="month:N",
            y="premium:Q"
        ),
        use_container_width=True
    )

st.caption(f"© {datetime.now().year} Aegis Home – Fast Demo")
