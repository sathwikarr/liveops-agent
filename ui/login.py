import streamlit as st

st.set_page_config(page_title="Login")

def login_page():
    st.title("🔐 LiveOps Agent - Login")

    # Already logged in
    if "username" in st.session_state:
        st.success(f"Already logged in as {st.session_state['username']}")
        if st.button("Go to Dashboard"):
            st.switch_page("pages/dashboard.py")  # Correct path to dashboard.py
        return

    # Login form
    username = st.text_input("Username", key="login_username")
    if st.button("Log In"):
        if username:
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.switch_page("pages/dashboard.py")  # Correct path to dashboard.py
        else:
            st.error("Please enter a username")


login_page()