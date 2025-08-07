import streamlit as st
import os

def login_page():
    st.title("🔐 LiveOps Agent - Login")

    if "username" in st.session_state:
        st.success(f"Already logged in as {st.session_state['username']}")
        if st.button("Go to Dashboard"):
            st.switch_page("Dashboard.py")
        return

    username = st.text_input("Username", key="login_username")
    if st.button("Log In"):
        if username:
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.switch_page("Dashboard.py")
        else:
            st.error("Please enter a username")