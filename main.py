import streamlit as st

def main():
    st.title("Main Page")
    
    # Navigation links
    page = st.sidebar.selectbox("Choose a page", ["Home", "Regression Analysis Tool", "Data Preprocessing"])

    if page == "Home":
        st.write("Welcome to the Main Page")
        st.write("Please select a page from the sidebar to get started.")
    elif page == "Regression Analysis Tool":
        st.write("Redirecting to Regression Analysis Tool...")
        # Redirect to Regression Analysis Tool page
        st.experimental_set_query_params(page="regression_tool")
    elif page == "Data Preprocessing":
        st.write("Redirecting to Data Preprocessing Page...")
        # Redirect to Data Preprocessing page
        st.experimental_set_query_params(page="data_preprocessing")

if __name__ == "__main__":
    main()
