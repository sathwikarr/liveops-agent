import streamlit as st

st.set_page_config(page_title="Login")

def login_page():
    st.title("🔐 LiveOps Agent - Login")

    # Already logged in
    if "username" in st.session_state:
        st.success(f"Already logged in as {st.session_state['username']}")
        if st.button("Go to Dashboard"):
            st.switch_page("pages/dashboard.py")  # Relative to app.py
        return

    # Login form
    username = st.text_input("Username", key="login_username")
    if st.button("Log In"):
        if username.strip():  # Check for non-empty username
            st.session_state["username"] = username.strip()
            st.success(f"Welcome, {username.strip()}! Redirecting...")
            st.switch_page("pages/dashboard.py")  # Relative to app.py
        else:
            st.error("Please enter a valid username")

if __name__ == "__main__":
    login_page()