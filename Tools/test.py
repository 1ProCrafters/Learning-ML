import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
from sklearn.pipeline import make_pipeline
matplotlib.use('TkAgg')  # Set the backend
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

TRAIN_DATA_COLOR = 'blue'
TEST_DATA_COLOR = 'red'
REGRESSION_LINE_COLOR = 'green'

def get_user_input():
    """Prompts the user for the CSV file name and returns it."""
    full_path = os.path.realpath(__file__)    
    csv_file = (os.path.dirname(full_path) + "\\" + input("Enter the name of the CSV file: ") + ".csv").replace("\\", "\\\\")
    return csv_file

def load_data(csv_file):
    """Loads data from the CSV file and returns the features (X) and target (y)."""
    try:
        data = pd.read_csv(csv_file)
        X = data.drop('target', axis=1)  # Assuming 'target' is the dependent variable
        y = data.drop(data.columns[:-1], axis=1)
        y = data['target']
        print(len(X), len(y))
        return X, y
    except FileNotFoundError:
        print("Error: CSV file not found. Please try again.")
        exit(1)

def run_regression_model(X, y, model_name, model_func, metric_func, visualize_data, is_classification=False):
    """
    Performs the specified regression model and returns the model, accuracy metric, and model name.

    Args:
        X: Features (independent variables)
        y: Target variable (dependent variable)
        model_name: Name of the regression model (for informative output)
        model_func: Function that implements the regression model (e.g., LinearRegression)
        metric_func: Function to calculate the accuracy metric (e.g., r2_score, mean_squared_error)
        is_classification (Optional): Boolean flag indicating if it's a classification model (affects metric)
        visualize_data (Optional): Boolean flag indicating if data visualization is desired

    Returns:
        model: The trained regression model
        accuracy: The accuracy metric value (R-squared for regression, other metrics for classification)
        model_name: The name of the model
        equation: The equation representing the model (if applicable)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

    model = model_func()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification:
        # Use appropriate metric for classification models (e.g., accuracy)
        accuracy = metric_func(y_test, y_pred)
    else:
        # Choose the desired metric based on user input
        accuracy = metric_func(y_test, y_pred)

    equation = ""
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coef = model.coef_[0]  # Assuming single feature for simplicity
        intercept = model.intercept_
        equation = f"y = {coef:.4f}x + {intercept:.4f}"  # Format equation string

    equation = equation.replace("y = ", "")

    if visualize_data and equation != "":
        model_str = str(model).split('(')[0]  # Extract only the model name without parentheses
        if model_str == "LinearRegression":
            if X_train.shape[0]!= y_train.shape[0]:
                raise ValueError("X_train and y_train must have the same size")
            plt.scatter(X_train, y_train, color=TRAIN_DATA_COLOR, label='Train Data')            
            plt.scatter(X_test, y_test, color=TEST_DATA_COLOR, label='Test Data')
            plt.plot(X_test, y_pred, color=REGRESSION_LINE_COLOR, label='Regression Line')
            plt.xlabel('Feature')
            plt.ylabel('Target Variable')
            plt.title(f"{model_name} Visualization")
            plt.legend()
            plot_name = f"{model_name.replace(' ', '_')}_visualization.png"  # Unique name for the plot
            plt.savefig(plot_name)  # Save plot as image
            plt.show()  # Show the plot
        else:
            print(f"{model_str}")

    return model, accuracy, model_name, equation

def run_simple_linear_regression(X, y, visualize_data, metric="r2"):
    """Performs Simple Linear Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    # Selecting the first column of X as the feature for simple linear regression
    X_feature = X.values.reshape(-1, 1)
    return run_regression_model(X_feature, y, "Simple Linear Regression", LinearRegression, metric_func, visualize_data)  # Default to R-squared

def run_multiple_linear_regression(X, y, visualize_data, metric="r2"):
    """Performs Multiple Linear Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, "Multiple Linear Regression", LinearRegression, metric_func, visualize_data)  # Default to R-squared

def run_polynomial_regression(X, y, visualize_data, degree, metric="r2"):
    """Runs polynomial regression on the input data with the specified degree."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)
    model_func = lambda: make_pipeline(PolynomialFeatures(degree), LinearRegression())
    return run_regression_model(X_poly, y, f"Polynomial Regression (Degree: {degree})", model_func, metric_func, visualize_data)

def run_ridge_regression(X, y, visualize_data, alpha, metric="r2"):
    """Performs Ridge Regression with the specified alpha using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, f"Ridge Regression (Alpha: {alpha})", Ridge, metric_func, visualize_data)  # Default to R-squared

def run_lasso_regression(X, y, visualize_data, alpha, metric="r2"):
    """Performs Lasso Regression with the specified alpha using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, f"Lasso Regression (Alpha: {alpha})", Lasso, metric_func, visualize_data)  # Default to R-squared

def run_svr(X, y, visualize_data, metric="r2"):
    """Performs Support Vector Regression (SVR) using the run_regression_model function.

    Args:
        metric (str, optional): The accuracy metric to use (default: "r2").
            Can be "r2" for R-squared or "mse" for mean squared error.
    """
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, "Support Vector Regression (SVR)", SVR, metric_func, visualize_data)

def run_decision_tree_regression(X, y, visualize_data, metric="r2"):
    """Performs Decision Tree Regression using the run_regression_model function.

    Args:
        metric (str, optional): The accuracy metric to use (default: "r2").
            Can be "r2" for R-squared or "mse" for mean squared error.
    """
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, "Decision Tree Regression", DecisionTreeRegressor, metric_func, visualize_data)

def run_random_forest_regression(X, y, visualize_data, metric="r2"):
    """Performs Random Forest Regression using the run_regression_model function.

    Args:
        metric (str, optional): The accuracy metric to use (default: "r2").
            Can be "r2" for R-squared or "mse" for mean squared error.
    """
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, "Random Forest Regression", RandomForestRegressor, metric_func, visualize_data)

def main():
    """
    Main function that prompts the user, loads data, runs regression models,
    and prints the results.
    """
    
    csv_file = get_user_input()
    X, y = load_data(csv_file)
    print("X: " + str(X))
    print("y: " + str(y))

    # Choose the desired metric (default to R-squared)
    metric = input("Enter the desired accuracy metric (r2 or mse): ")
    metric = metric.lower()  # Convert to lowercase for case-insensitive input
    
    visualize = input("Do you want to visualize the data? (Yes or No): ")
    visualize_data = visualize.lower() == "yes"

    # Run different regression models
    models = [
        run_simple_linear_regression(X, y, visualize_data, metric=metric),
        run_multiple_linear_regression(X, y, visualize_data, metric=metric),
        run_polynomial_regression(X.copy(), y, visualize_data, degree=2, metric=metric),  # Copy X to avoid modifying original data
        # run_polynomial_regression(X.copy(), y, visualize_data, degree=3, metric=metric),
        run_ridge_regression(X, y, visualize_data, alpha=1.0, metric=metric),
        run_lasso_regression(X, y, visualize_data, alpha=0.1, metric=metric),
        run_svr(X, y, visualize_data, metric=metric),
        run_decision_tree_regression(X, y, visualize_data, metric=metric),
        run_random_forest_regression(X, y, visualize_data, metric=metric),
    ]

    # Find the model with the highest accuracy
    best_model = None
    best_accuracy = float('-inf')
    best_equation = None
    for model, accuracy, model_name, equation in models:
        if not equation:
            equation = "Equation not available for this model."
        else:
            equation = "Equation: " + equation
        print(f"{model_name}: Accuracy = {accuracy:.4f}; {equation}")
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
            best_equation = equation

    if best_model:
        model = re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', str(best_model.__class__.__name__))  # Split a “caps” delimited string into a string?
        print(f"\nBest Model: '{model}' with Accuracy = {best_accuracy:.4f} and {best_equation}")
    else:
        print("\nNo models were run successfully.")

if __name__ == "__main__":
    main()
