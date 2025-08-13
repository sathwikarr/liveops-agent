import streamlit as st
from ui.login import login_page
from agent.explain import explain_anomaly
from agent.detect import read_latest_data, zscore_anomaly_detection

st.set_page_config(page_title="LiveOps Agent")

# Run the login page
login_page()