import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    st.title("Regression Analysis Tool")
    st.write("Upload your dataset and choose the regression algorithm to get started.")

    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.experimental_rerun()

    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        if st.sidebar.button("Generate Comparative Report"):
            st.session_state.page = "make_report"
            st.experimental_rerun()
            
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        independent_columns = st.sidebar.multiselect("Select independent (predictor) variables:", options=df.columns.tolist())
        dependent_column = st.sidebar.selectbox("Select dependent (target) variable:", options=df.columns.tolist())

        if not independent_columns:
            st.warning("Please select at least one independent (predictor) variable.")
            return

        if not dependent_column:
            st.warning("Please select the dependent (target) variable.")
            return

        categorical_option = st.sidebar.radio(
            "Do you have any categorical variables?",
            ('No', 'Yes')
        )

        if categorical_option == 'Yes':
            categorical_columns = st.sidebar.multiselect("Select categorical columns:", options=independent_columns)
        else:
            categorical_columns = []

        # Function to handle categorical variables
        def handle_categorical(df, categorical_columns):
            df_processed = df.copy()
            for col in categorical_columns:
                df_processed[col] = pd.Categorical(df_processed[col]).codes
            return df_processed

        # Function to split the dataset
        def split(df, test_size, dependent_column):
            X = df.drop(dependent_column, axis=1)
            y = df[dependent_column]
            return train_test_split(X, y, test_size=test_size, random_state=42)

        # Function to show the split
        def show_split(x_train, x_test, test_size):
            st.write(f"Training set: {len(x_train)} samples")
            st.write(f"Test set: {len(x_test)} samples ({test_size*100}% of the data)")

        # Function to calculate adjusted R2 score
        def adjusted_r2(r2, n, k):
            if n <= k + 1:
                return float('nan')  # Adjusted R² is undefined
            return 1 - (((1 - r2) * (n - 1)) / (n - k - 1))

        # Store results for the report
        regression_results = {}

        algo = st.sidebar.selectbox("Select Regression Algorithm:", 
                            ['Linear Regression', 
                             'Polynomial Regression', 
                             'Support Vector Regression (SVR)', 
                             'Decision Tree Regression', 
                             'Random Forest Regression'])

        df_processed = handle_categorical(df, categorical_columns)

        if algo == 'Linear Regression':
            st.write(f"<h2><b><font>Linear Regression</b></h2>", unsafe_allow_html=True)
            test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.01, key="test_size_lr")

            x_train, x_test, y_train, y_test = split(df_processed, test_size, dependent_column)
            show_split(x_train, x_test, test_size)

            lr = LinearRegression()
            lr.fit(x_train[independent_columns], y_train)
            y_predictions = lr.predict(x_test[independent_columns])

            st.header("Linear Regression Results: ")

            c1, c2, c3 = st.columns(3)

            with c1:
                r2_scoree = r2_score(y_test, y_predictions)
                st.write(f"<b>R2 score is: <br><mark> {r2_scoree:.6f} </mark></b>", unsafe_allow_html=True)
            with c2:
                adjusted_r2_score = adjusted_r2(r2_scoree, len(x_test), len(independent_columns))
                st.write(f"<b>Adjusted R2 score is: <mark> {adjusted_r2_score:.6f} </mark></b>", unsafe_allow_html=True)
            with c3:
                mean_squared_error_ = mean_squared_error(y_test, y_predictions)
                st.write(f"<b>Root mean squared error is: <mark> {np.sqrt(mean_squared_error_):.6f} </mark></b>", unsafe_allow_html=True)

            st.subheader("Regression Plot")
            e = st.expander("")
            x_bins = e.number_input("x_bins", 10, 100, 10)
            fig, ax = plt.subplots()
            sns.regplot(x=y_test, y=y_predictions, robust=True, color='blue', x_bins=x_bins, ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            st.subheader("Residual Plot")
            e = st.expander("")     
            fig, ax = plt.subplots()
            sns.residplot(x=y_test, y=y_predictions, robust=True, color='blue', ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            if st.button("Save to Report", key="save_lr"):
                regression_results['Linear Regression'] = {
                    'R2 Score': r2_scoree,
                    'Adjusted R2 Score': adjusted_r2_score,
                    'Root Mean Squared Error': np.sqrt(mean_squared_error_),
                    'Regression Plot': fig,
                    'Residual Plot': fig
                }

        if algo == 'Polynomial Regression':
            st.write(f"<h2><b><font>Polynomial Regression</b></h2>", unsafe_allow_html=True)
            degree = st.sidebar.slider("Degree of the polynomial features:", 2, 5, 2, key="degree_pr")
            test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.01, key="test_size_pr")

            x_train, x_test, y_train, y_test = split(df_processed, test_size, dependent_column)
            show_split(x_train, x_test, test_size)

            poly_features = PolynomialFeatures(degree=degree)
            x_train_poly = poly_features.fit_transform(x_train[independent_columns])
            x_test_poly = poly_features.fit_transform(x_test[independent_columns])

            pr = LinearRegression()
            pr.fit(x_train_poly, y_train)
            y_predictions = pr.predict(x_test_poly)

            st.header("Polynomial Regression Results: ")

            c1, c2, c3 = st.columns(3)

            with c1:
                r2_scoree = r2_score(y_test, y_predictions)
                st.write(f"<b>R2 score is: <mark> {r2_scoree:.6f} </mark></b>", unsafe_allow_html=True)
            with c2:
                adjusted_r2_score = adjusted_r2(r2_scoree, len(x_test), len(independent_columns))
                st.write(f"<b>Adjusted R2 score is: <mark> {adjusted_r2_score:.6f} </mark></b>", unsafe_allow_html=True)
            with c3:
                mean_squared_error_ = mean_squared_error(y_test, y_predictions)
                st.write(f"<b>Root mean squared error is: <mark> {np.sqrt(mean_squared_error_):.6f} </mark></b>", unsafe_allow_html=True)

            st.subheader("Regression Plot")
            e = st.expander("")
            x_bins = e.number_input("x_bins", 10, 100, 10)
            fig, ax = plt.subplots()
            sns.regplot(x=y_test, y=y_predictions, robust=True, color='blue', x_bins=x_bins, ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            st.subheader("Residual Plot")
            e = st.expander("")     
            fig, ax = plt.subplots()
            sns.residplot(x=y_test, y=y_predictions, robust=True, color='blue', ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            if st.button("Save to Report", key="save_pr"):
                regression_results['Polynomial Regression'] = {
                    'R2 Score': r2_scoree,
                    'Adjusted R2 Score': adjusted_r2_score,
                    'Root Mean Squared Error': np.sqrt(mean_squared_error_),
                    'Regression Plot': fig,
                    'Residual Plot': fig
                }

        if algo == 'Support Vector Regression (SVR)':
            st.write(f"<h2><b><font>Support Vector Regression (SVR)</b></h2>", unsafe_allow_html=True)
            kernel = st.sidebar.selectbox("Select SVR kernel:", ['linear', 'poly', 'rbf', 'sigmoid'], key="kernel_svr")
            test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.01, key="test_size_svr")

            x_train, x_test, y_train, y_test = split(df_processed, test_size, dependent_column)
            show_split(x_train, x_test, test_size)

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train[independent_columns])
            x_test_scaled = scaler.transform(x_test[independent_columns])

            svr = SVR(kernel=kernel)
            svr.fit(x_train_scaled, y_train)
            y_predictions = svr.predict(x_test_scaled)

            st.header("SVR Results: ")

            c1, c2, c3 = st.columns(3)

            with c1:
                r2_scoree = r2_score(y_test, y_predictions)
                st.write(f"<b>R2 score is: <mark> {r2_scoree:.6f} </mark></b>", unsafe_allow_html=True)
            with c2:
                adjusted_r2_score = adjusted_r2(r2_scoree, len(x_test), len(independent_columns))
                st.write(f"<b>Adjusted R2 score is: <mark> {adjusted_r2_score:.6f} </mark></b>", unsafe_allow_html=True)
            with c3:
                mean_squared_error_ = mean_squared_error(y_test, y_predictions)
                st.write(f"<b>Root mean squared error is: <mark> {np.sqrt(mean_squared_error_):.6f} </mark></b>", unsafe_allow_html=True)

            st.subheader("Regression Plot")
            e = st.expander("")
            x_bins = e.number_input("x_bins", 10, 100, 10)
            fig, ax = plt.subplots()
            sns.regplot(x=y_test, y=y_predictions, robust=True, color='blue', x_bins=x_bins, ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            st.subheader("Residual Plot")
            e = st.expander("")     
            fig, ax = plt.subplots()
            sns.residplot(x=y_test, y=y_predictions, robust=True, color='blue', ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            if st.button("Save to Report", key="save_svr"):
                regression_results['Support Vector Regression'] = {
                    'R2 Score': r2_scoree,
                    'Adjusted R2 Score': adjusted_r2_score,
                    'Root Mean Squared Error': np.sqrt(mean_squared_error_),
                    'Regression Plot': fig,
                    'Residual Plot': fig
                }

        if algo == 'Decision Tree Regression':
            st.write(f"<h2><b><font>Decision Tree Regression</b></h2>", unsafe_allow_html=True)
            test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.01, key="test_size_dtr")

            x_train, x_test, y_train, y_test = split(df_processed, test_size, dependent_column)
            show_split(x_train, x_test, test_size)

            dtr = DecisionTreeRegressor(random_state=42)
            dtr.fit(x_train[independent_columns], y_train)
            y_predictions = dtr.predict(x_test[independent_columns])

            st.header("Decision Tree Regression Results: ")

            c1, c2, c3 = st.columns(3)

            with c1:
                r2_scoree = r2_score(y_test, y_predictions)
                st.write(f"<b>R2 score is: <mark> {r2_scoree:.6f} </mark></b>", unsafe_allow_html=True)
            with c2:
                adjusted_r2_score = adjusted_r2(r2_scoree, len(x_test), len(independent_columns))
                st.write(f"<b>Adjusted R2 score is: <mark> {adjusted_r2_score:.6f} </mark></b>", unsafe_allow_html=True)
            with c3:
                mean_squared_error_ = mean_squared_error(y_test, y_predictions)
                st.write(f"<b>Root mean squared error is: <mark> {np.sqrt(mean_squared_error_):.6f} </mark></b>", unsafe_allow_html=True)

            st.subheader("Decision Tree Plot")
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_tree(dtr, feature_names=independent_columns, filled=True, ax=ax)
            st.pyplot(fig)

            if st.button("Save to Report", key="save_dtr"):
                regression_results['Decision Tree Regression'] = {
                    'R2 Score': r2_scoree,
                    'Adjusted R2 Score': adjusted_r2_score,
                    'Root Mean Squared Error': np.sqrt(mean_squared_error_),
                    'Decision Tree Plot': fig
                }

        if algo == 'Random Forest Regression':
            st.write(f"<h2><b><font>Random Forest Regression</b></h2>", unsafe_allow_html=True)
            n_estimators = st.sidebar.slider("Number of trees in the forest:", 10, 100, 10, key="n_estimators_rfr")
            test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.01, key="test_size_rfr")

            x_train, x_test, y_train, y_test = split(df_processed, test_size, dependent_column)
            show_split(x_train, x_test, test_size)

            rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            rfr.fit(x_train[independent_columns], y_train)
            y_predictions = rfr.predict(x_test[independent_columns])

            st.header("Random Forest Regression Results: ")

            c1, c2, c3 = st.columns(3)

            with c1:
                r2_scoree = r2_score(y_test, y_predictions)
                st.write(f"<b>R2 score is: <mark> {r2_scoree:.6f} </mark></b>", unsafe_allow_html=True)
            with c2:
                adjusted_r2_score = adjusted_r2(r2_scoree, len(x_test), len(independent_columns))
                st.write(f"<b>Adjusted R2 score is: <mark> {adjusted_r2_score:.6f} </mark></b>", unsafe_allow_html=True)
            with c3:
                mean_squared_error_ = mean_squared_error(y_test, y_predictions)
                st.write(f"<b>Root mean squared error is: <mark> {np.sqrt(mean_squared_error_):.6f} </mark></b>", unsafe_allow_html=True)

            st.subheader("Regression Plot")
            e = st.expander("")
            x_bins = e.number_input("x_bins", 10, 100, 10)
            fig, ax = plt.subplots()
            sns.regplot(x=y_test, y=y_predictions, robust=True, color='blue', x_bins=x_bins, ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            st.subheader("Residual Plot")
            e = st.expander("")     
            fig, ax = plt.subplots()
            sns.residplot(x=y_test, y=y_predictions, robust=True, color='blue', ax=ax)
            e.pyplot(fig=fig, clear_figure=None)

            if st.button("Save to Report", key="save_rfr"):
                regression_results['Random Forest Regression'] = {
                    'R2 Score': r2_scoree,
                    'Adjusted R2 Score': adjusted_r2_score,
                    'Root Mean Squared Error': np.sqrt(mean_squared_error_),
                    'Regression Plot': fig,
                    'Residual Plot': fig
                }

        # Save the regression results to session state
        st.session_state['regression_results'] = regression_results

if __name__ == "__main__":
    main()
