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
    csv_file = (os.path.dirname(full_path) + "\\" + (input("Enter the name of the CSV file (Default: Data): ") or "Data") + ".csv").replace("\\", "\\\\")
    return csv_file

def load_data(csv_file):
    """Loads data from the CSV file and returns the features (X) and target (y)."""
    try:
        data = pd.read_csv(csv_file)
        X = data.drop('target', axis=1)  # Assuming 'target' is the dependent variable
        y = data['target']
        return X, y
    except FileNotFoundError:
        print("Error: CSV file not found. Please try again.")
        exit(1)

def simplify(equation):
    # Extract coefficients and constant term
    pattern = r"y = (-?\d*)\*x ([+-] \d+)?"
    match = re.match(pattern, equation)

    if match:
        coefficient = int(match.group(1))
        constant = int(match.group(2).replace(" ", "")) if match.group(2) else 0

        # Simplify the equation
        simplified_equation = f"y = {coefficient}*x"
        if constant != 0:
            simplified_equation += f" + {constant}"

        print(simplified_equation)
    else:
        print("Equation format not recognized.")

def generate_simplified_equation(model, feature_names=None, threshold=1e-4):
    """Generates a simplified human-readable equation string from the model's coefficients and intercept."""
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        coef = model.coef_
        intercept = model.intercept_
        terms = []
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(len(coef))]
        for i, c in enumerate(coef):
            if abs(c) > threshold:
                terms.append(f"{c:.4f}*{feature_names[i]}")
        equation = " + ".join(terms)
        if abs(intercept) > threshold:
            equation = f"{intercept:.4f} + " + equation
        return equation
    return "Equation not available for this model."

def run_regression_model(X, y, model_name, model_func, metric_func, visualize_data, feature_names=None):
    """
    Performs the specified regression model and returns the model, accuracy metric, and model name.

    Args:
        X: Features (independent variables)
        y: Target variable (dependent variable)
        model_name: Name of the regression model (for informative output)
        model_func: Function that implements the regression model (e.g., LinearRegression)
        metric_func: Function to calculate the accuracy metric (e.g., r2_score, mean_squared_error)
        visualize_data (Optional): Boolean flag indicating if data visualization is desired
        feature_names (Optional): List of feature names for the equation string.

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

    accuracy = metric_func(y_test, y_pred)

    equation = generate_simplified_equation(model, feature_names)

    if visualize_data:
        plt.scatter(X_train.iloc[:, 0], y_train, color=TRAIN_DATA_COLOR, label='Train Data')
        plt.scatter(X_test.iloc[:, 0], y_test, color=TEST_DATA_COLOR, label='Test Data')
        plt.plot(X_test.iloc[:, 0], y_pred, color=REGRESSION_LINE_COLOR, label='Regression Line')
        plt.xlabel('Feature')
        plt.ylabel('Target Variable')
        plt.title(f"{model_name} Visualization")
        plt.legend()
        plot_name = f"{model_name.replace(' ', '_')}_visualization.png"  # Unique name for the plot
        plt.savefig(plot_name)  # Save plot as image
        plt.show()  # Show the plot

    return model, accuracy, model_name, equation

def run_simple_linear_regression(X, y, visualize_data, metric="r2"):
    """Performs Simple Linear Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    feature_names = [X.columns[0]]
    X_feature = X.iloc[:, 0].values.reshape(-1, 1)
    return run_regression_model(pd.DataFrame(X_feature, columns=feature_names), y, "Simple Linear Regression", LinearRegression, metric_func, visualize_data, feature_names)

def run_multiple_linear_regression(X, y, visualize_data, metric="r2"):
    """Performs Multiple Linear Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    feature_names = X.columns
    return run_regression_model(X, y, "Multiple Linear Regression", LinearRegression, metric_func, visualize_data, feature_names)

def run_polynomial_regression(X, y, visualize_data, degree, metric="r2"):
    """Runs polynomial regression on the input data with the specified degree."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    feature_names = polynomial_features.get_feature_names_out(X.columns)
    equation = generate_simplified_equation(model, feature_names)

    if visualize_data:
        plt.scatter(X, y, color=TRAIN_DATA_COLOR, label='Data')
        X_seq = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        plt.plot(X_seq, model.predict(polynomial_features.transform(X_seq)), color=REGRESSION_LINE_COLOR, label='Regression Line')
        plt.xlabel('Feature')
        plt.ylabel('Target Variable')
        plt.title(f"Polynomial Regression (Degree: {degree}) Visualization")
        plt.legend()
        plot_name = f"Polynomial_Regression_Degree_{degree}_visualization.png"  # Unique name for the plot
        plt.savefig(plot_name)  # Save plot as image
        plt.show()  # Show the plot

    return model, metric_func(y, y_pred), f"Polynomial Regression (Degree: {degree})", equation

def run_ridge_regression(X, y, visualize_data, alpha, metric="r2"):
    """Performs Ridge Regression with the specified alpha using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, f"Ridge Regression (Alpha: {alpha})", Ridge, metric_func, visualize_data)

def run_lasso_regression(X, y, visualize_data, alpha, metric="r2"):
    """Performs Lasso Regression with the specified alpha using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, f"Lasso Regression (Alpha: {alpha})", Lasso, metric_func, visualize_data)

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
    
    condition = "greater_than"
    
    operators = {
        "greater_than": ">",
        "less_than": "<",
        "equal_to": "=="
    }
    
    csv_file = get_user_input()
    X, y = load_data(csv_file)

    # Choose the desired metric (default to R-squared)
    metric = input("Enter the desired accuracy metric (r2 or Default: mse): ")
    metric = metric.lower()  # Convert to lowercase for case-insensitive input
    
    visualize = input("Do you want to visualize the data? (Yes or Default: No): " or "No").lower()

    visualize_data = visualize == "yes"

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
    best_accuracy = None
    best_equation = None
    best_model_name = None
    
    if (metric == "r2"):
        condition = "greater_than"
    elif (metric == "mse" or metric == ""):
        condition = "less_than"
    else:
        raise ValueError("Invalid metric entered.")

    for model, accuracy, model_name, equation in models:
        if not equation:
            equation = "Equation not available for this model."
        else:
            equation = "Equation: " + equation
        print(f"{model_name}: Accuracy = {accuracy:.4f}; {equation}")
        
        if best_accuracy == None: best_accuracy = accuracy        
        if eval(str(accuracy) + str(operators[condition]) + str(best_accuracy)):
            best_model = model
            best_accuracy = accuracy
            best_equation = equation
            best_model_name = model_name

    if best_model:
        print(f"\nBest Model: '{best_model_name}' with Accuracy = {best_accuracy:.4f} and {best_equation}")
    else:
        print("\nNo models were run successfully.")

if __name__ == "__main__":
    main()
