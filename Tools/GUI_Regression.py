import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
from sklearn.pipeline import make_pipeline
from tkinter import filedialog, ttk
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from io import BytesIO, StringIO
import base64

matplotlib.use('Agg')

TRAIN_DATA_COLOR = 'blue'
TEST_DATA_COLOR = 'red'
REGRESSION_LINE_COLOR = 'green'

def load_data(csv_file):
    """Loads data from the CSV file and returns the features (X) and target (y)."""
    try:
        data = pd.read_csv(csv_file)
        X = data.drop('target', axis=1)  # Assuming 'target' is the dependent variable
        y = data['target']
        return X, y, data
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
    Performs the specified regression model and returns the model, accuracy metric, model name, equation, and plot data.

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
        plot_data: Base64-encoded image data of the plot
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

    model = model_func()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = metric_func(y_test, y_pred)

    equation = generate_simplified_equation(model, feature_names)

    fig, ax = plt.subplots()
    ax.scatter(X_train.iloc[:, 0], y_train, color=TRAIN_DATA_COLOR, label='Train Data')
    ax.scatter(X_test.iloc[:, 0], y_test, color=TEST_DATA_COLOR, label='Test Data')
    ax.plot(X_test.iloc[:, 0], y_pred, color=REGRESSION_LINE_COLOR, label='Regression Line')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target Variable')
    ax.set_title(f"{model_name} Visualization")
    ax.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

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

    fig, ax = plt.subplots()
    ax.scatter(X, y, color=TRAIN_DATA_COLOR, label='Data')
    X_seq = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    ax.plot(X_seq, model.predict(polynomial_features.transform(X_seq)), color=REGRESSION_LINE_COLOR, label='Regression Line')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target Variable')
    ax.set_title(f"Polynomial Regression (Degree: {degree}) Visualization")
    ax.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

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
            display_csv(file_path)

    def display_csv(file_path):
        try:
            data = pd.read_csv(file_path)
            text_csv.delete(1.0, tk.END)
            text_csv.insert(tk.END, data.to_csv(index=False))
        except Exception as e:
            text_csv.delete(1.0, tk.END)
            text_csv.insert(tk.END, f"Error loading CSV file: {e}")

    def save_csv():
        file_path = entry_file_path.get()
        csv_data = text_csv.get(1.0, tk.END)
        try:
            df = pd.read_csv(StringIO(csv_data))
            df.to_csv(file_path, index=False)
            label_status.config(text="CSV file saved successfully!", fg="green")
        except Exception as e:
            label_status.config(text=f"Error saving CSV file: {e}", fg="red")

    def run_models():
        csv_file = entry_file_path.get()
        X, y, data = load_data(csv_file)

        metric = combo_metric.get().lower()

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
            label_result = tk.Label(tab_summary, text=result)
            label_result.pack()
            
            img = tk.PhotoImage(data=plot_data)
            label_image = tk.Label(tab_visualizations, image=img)
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

    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, fill=tk.BOTH)

    # Tab 1: File selection and options
    tab_file = ttk.Frame(notebook)
    notebook.add(tab_file, text="File Selection and Options")

    frame_file = tk.Frame(tab_file)
    frame_file.pack(pady=10)
    tk.Label(frame_file, text="CSV File:").pack(side=tk.LEFT)
    entry_file_path = tk.Entry(frame_file, width=50)
    entry_file_path.pack(side=tk.LEFT, padx=5)
    btn_browse = tk.Button(frame_file, text="Browse", command=load_file)
    btn_browse.pack(side=tk.LEFT)

    frame_options = tk.Frame(tab_file)
    frame_options.pack(pady=10)
    tk.Label(frame_options, text="Accuracy Metric:").pack(side=tk.LEFT)
    combo_metric = ttk.Combobox(frame_options, values=["r2", "mse"], state="readonly")
    combo_metric.set("mse")
    combo_metric.pack(side=tk.LEFT, padx=5)

    btn_run = tk.Button(tab_file, text="Run Models", command=run_models)
    btn_run.pack(pady=10)

    # Tab 2: Edit CSV
    tab_edit_csv = ttk.Frame(notebook)
    notebook.add(tab_edit_csv, text="Edit CSV")

    text_csv = ScrolledText(tab_edit_csv, wrap=tk.WORD)
    text_csv.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    frame_save = tk.Frame(tab_edit_csv)
    frame_save.pack(pady=10)
    btn_save_csv = tk.Button(frame_save, text="Save CSV", command=save_csv)
    btn_save_csv.pack()

    label_status = tk.Label(tab_edit_csv, text="", fg="red")
    label_status.pack(pady=5)

    # Tab 3: Summary
    tab_summary = ttk.Frame(notebook)
    notebook.add(tab_summary, text="Summary")

    label_best_model = tk.Label(tab_summary, text="", fg="blue")
    label_best_model.pack(pady=10)

    # Tab 4: Visualizations
    tab_visualizations = ttk.Frame(notebook)
    notebook.add(tab_visualizations, text="Visualizations")

    root.mainloop()

if __name__ == "__main__":
    main()
