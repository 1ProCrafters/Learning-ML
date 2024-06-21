import streamlit as st

def main():
    st.title("Data Preprocessing Page")
    st.write("This page is for data preprocessing.")

    # Add a button to return to the main page
    if st.button("Return to Main Page"):
        st.experimental_set_query_params(page="main")

if __name__ == "__main__":
    main()
