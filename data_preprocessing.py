import pandas as pd
import streamlit as st

def main():
    st.title("Data Preprocessing Tool")
    st.write("Upload your dataset and preprocess it before applying regression.")

    if st.button("Return to Main Page"):
        st.session_state.page = "main"

    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        st.subheader("Data Summary:")
        st.write(df.describe())

        st.subheader("Data Information:")
        st.write(df.info())

        st.subheader("Columns in Dataset:")
        st.write(df.columns.tolist())

        st.subheader("Missing Values:")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])

        st.subheader("Data Preprocessing:")
        selected_columns = st.multiselect("Select columns to preprocess:", options=df.columns.tolist())

        if st.button("Preprocess Selected Columns"):
            if selected_columns:
                processed_df = preprocess_data(df[selected_columns])
                st.write("Processed Dataset Preview:")
                st.write(processed_df.head())
            else:
                st.warning("Please select columns to preprocess.")

def preprocess_data(df):
    # Example preprocessing steps (replace with your own logic)
    # Currently, it just fills missing values with mean for numeric columns
    for col in df.select_dtypes(include='number'):
        df[col].fillna(df[col].mean(), inplace=True)
    
    # Encode categorical variables if needed
    for col in df.select_dtypes(include='object'):
        df[col] = pd.Categorical(df[col]).codes
    
    return df

if __name__ == "__main__":
    main()
