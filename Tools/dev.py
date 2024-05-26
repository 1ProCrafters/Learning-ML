import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import base64
from io import BytesIO
import tkinter as tk
from tkinter import filedialog, ttk

# Constants
TRAIN_DATA_COLOR = 'blue'
TEST_DATA_COLOR = 'red'
REGRESSION_LINE_COLOR = 'green'

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

def run_regression_model(X, y, model_name, model_func, metric_func, feature_names=None):
    """
    Performs the specified regression model and returns the model, accuracy metric, and model name.

    Args:
        X: Features (independent variables)
        y: Target variable (dependent variable)
        model_name: Name of the regression model (for informative output)
        model_func: Function that implements the regression model (e.g., LinearRegression)
        metric_func: Function to calculate the accuracy metric (e.g., r2_score, mean_squared_error)
        feature_names (Optional): List of feature names for the equation string.

    Returns:
        model: The trained regression model
        accuracy: The accuracy metric value (R-squared for regression, other metrics for classification)
        model_name: The name of the model
        equation: The equation representing the model (if applicable)
        plot_data: The plot image data as a base64 string
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

    model = model_func()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = metric_func(y_test, y_pred)
    equation = generate_simplified_equation(model, feature_names)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train.iloc[:, 0], y_train, color=TRAIN_DATA_COLOR, label='Train Data')
    plt.scatter(X_test.iloc[:, 0], y_test, color=TEST_DATA_COLOR, label='Test Data')
    plt.plot(X_test.iloc[:, 0], y_pred, color=REGRESSION_LINE_COLOR, label='Regression Line')
    plt.xlabel('Feature')
    plt.ylabel('Target Variable')
    plt.title(f"{model_name} Visualization")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return model, accuracy, model_name, equation, plot_data

def run_simple_linear_regression(X, y, metric="r2"):
    """Performs Simple Linear Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    feature_names = [X.columns[0]]
    X_feature = X.iloc[:, 0].values.reshape(-1, 1)
    return run_regression_model(pd.DataFrame(X_feature, columns=feature_names), y, "Simple Linear Regression", LinearRegression, metric_func, feature_names)

def run_multiple_linear_regression(X, y, metric="r2"):
    """Performs Multiple Linear Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    feature_names = X.columns
    return run_regression_model(X, y, "Multiple Linear Regression", LinearRegression, metric_func, feature_names)

def run_polynomial_regression(X, y, degree, metric="r2"):
    """Runs polynomial regression on the input data with the specified degree."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    feature_names = polynomial_features.get_feature_names_out(X.columns)
    equation = generate_simplified_equation(model, feature_names)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color=TRAIN_DATA_COLOR, label='Data')
    X_seq = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    plt.plot(X_seq, model.predict(polynomial_features.transform(X_seq)), color=REGRESSION_LINE_COLOR, label='Regression Line')
    plt.xlabel('Feature')
    plt.ylabel('Target Variable')
    plt.title(f"Polynomial Regression (Degree: {degree}) Visualization")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return model, metric_func(y, y_pred), f"Polynomial Regression (Degree: {degree})", equation, plot_data

def run_ridge_regression(X, y, alpha, metric="r2"):
    """Performs Ridge Regression with the specified alpha using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, f"Ridge Regression (Alpha: {alpha})", lambda: Ridge(alpha=alpha), metric_func)

def run_lasso_regression(X, y, alpha, metric="r2"):
    """Performs Lasso Regression with the specified alpha using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, f"Lasso Regression (Alpha: {alpha})", lambda: Lasso(alpha=alpha), metric_func)

def run_svr(X, y, metric="r2"):
    """Performs Support Vector Regression (SVR) using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, "Support Vector Regression (SVR)", SVR, metric_func)

def run_decision_tree_regression(X, y, metric="r2"):
    """Performs Decision Tree Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, "Decision Tree Regression", DecisionTreeRegressor, metric_func)

def run_random_forest_regression(X, y, metric="r2"):
    """Performs Random Forest Regression using the run_regression_model function."""
    metric_func = r2_score if metric == "r2" else mean_squared_error
    return run_regression_model(X, y, "Random Forest Regression", RandomForestRegressor, metric_func)

def main():
    def load_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            entry_file_path.delete(0, tk.END)
            entry_file_path.insert(0, file_path)

    def run_models():
        csv_file = entry_file_path.get()
        X, y = load_data(csv_file)

        metric = combo_metric.get().lower()
        visualize_data = chk_visualize_var.get()

        models = [
            run_simple_linear_regression(X, y, metric=metric),
            run_multiple_linear_regression(X, y, metric=metric),
            run_polynomial_regression(X.copy(), y, degree=2, metric=metric),
            run_ridge_regression(X, y, alpha=1.0, metric=metric),
            run_lasso_regression(X, y, alpha=0.1, metric=metric),
            run_svr(X, y, metric=metric),
            run_decision_tree_regression(X, y, metric=metric),
            run_random_forest_regression(X, y, metric=metric),
        ]

        best_model = None
        best_accuracy = None
        best_equation = None
        best_model_name = None

        condition = "greater_than" if metric == "r2" else "less_than"
        operators = {"greater_than": ">", "less_than": "<", "equal_to": "=="}

        for model, accuracy, model_name, equation, plot_data in models:
            if not equation:
                equation = "Equation not available for this model."
            else:
                equation = "Equation: " + equation
            result = f"{model_name}: Accuracy = {accuracy:.4f}; {equation}"
            label_result = tk.Label(frame_results, text=result)
            label_result.pack()
            
            img = tk.PhotoImage(data=plot_data)
            label_image = tk.Label(frame_results, image=img)
            label_image.image = img  # Keep a reference to avoid garbage collection
            label_image.pack()

            if best_accuracy is None:
                best_accuracy = accuracy
            if eval(f"{accuracy} {operators[condition]} {best_accuracy}"):
                best_model = model
                best_accuracy = accuracy
                best_equation = equation
                best_model_name = model_name

        if best_model:
            label_best_model.config(text=f"Best Model: '{best_model_name}' with Accuracy = {best_accuracy:.4f} and {best_equation}")
        else:
            label_best_model.config(text="No models were run successfully.")

    root = tk.Tk()
    root.title("Regression Model Comparison")

    # File selection frame
    frame_file = tk.Frame(root)
    frame_file.pack(pady=10)
    tk.Label(frame_file, text="CSV File:").pack(side=tk.LEFT)
    entry_file_path = tk.Entry(frame_file, width=50)
    entry_file_path.pack(side=tk.LEFT, padx=5)
    btn_browse = tk.Button(frame_file, text="Browse", command=load_file)
    btn_browse.pack(side=tk.LEFT)

    # Options frame
    frame_options = tk.Frame(root)
    frame_options.pack(pady=10)
    tk.Label(frame_options, text="Accuracy Metric:").pack(side=tk.LEFT)
    combo_metric = ttk.Combobox(frame_options, values=["r2", "mse"], state="readonly")
    combo_metric.set("mse")
    combo_metric.pack(side=tk.LEFT, padx=5)
    chk_visualize_var = tk.BooleanVar()
    chk_visualize = tk.Checkbutton(frame_options, text="Visualize Data", variable=chk_visualize_var)
    chk_visualize.pack(side=tk.LEFT, padx=5)

    # Run button
    btn_run = tk.Button(root, text="Run Models", command=run_models)
    btn_run.pack(pady=10)

    # Results frame
    frame_results = tk.Frame(root)
    frame_results.pack(pady=10, fill=tk.BOTH, expand=True)

    # Canvas and scrollbar
    canvas = tk.Canvas(frame_results)
    scrollbar = tk.Scrollbar(frame_results, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Best model label
    label_best_model = tk.Label(root, text="", fg="blue")
    label_best_model.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
