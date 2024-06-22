import streamlit as st
from regression_tool import main as regression_tool
from data_preprocessing import main as data_preprocessing

def main():
    if "page" not in st.session_state:
        st.session_state.page = "main"

    if st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "regression_tool":
        regression_tool()
    elif st.session_state.page == "data_preprocessing":
        data_preprocessing()

def main_page():
    st.title("Main Page")
    
    # Navigation links
    page = st.sidebar.selectbox("Choose a page", ["Home", "Regression Analysis Tool", "Data Preprocessing"])

    if page == "Home":
        st.write("Welcome to the Main Page")
        st.write("Please select a page from the sidebar to get started.")
    elif page == "Regression Analysis Tool":
        st.write("Redirecting to Regression Analysis Tool...")
        st.session_state.page = "regression_tool"
    elif page == "Data Preprocessing":
        st.write("Redirecting to Data Preprocessing Page...")
        st.session_state.page = "data_preprocessing"

if __name__ == "__main__":
    main()
