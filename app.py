import streamlit as st
from ui.login import login_page

st.set_page_config(page_title="LiveOps Agent")

# Run the login page
login_page()