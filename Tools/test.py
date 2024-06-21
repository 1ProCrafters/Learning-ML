import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText
import os
import pandas as pd
import webbrowser

# Global settings
TRAIN_DATA_COLOR = 'blue'
TEST_DATA_COLOR = 'red'
REGRESSION_LINE_COLOR = 'green'
VISUALIZATIONS_PER_ROW = 3

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y, data

def run_simple_linear_regression(X, y, metric):
    # Placeholder for running the model and generating results
    return None, 0.9, "Simple Linear Regression", "y = ax + b", None

def run_multiple_linear_regression(X, y, metric):
    # Placeholder for running the model and generating results
    return None, 0.85, "Multiple Linear Regression", "y = ax1 + bx2 + c", None

def run_polynomial_regression(X, y, degree, metric):
    # Placeholder for running the model and generating results
    return None, 0.92, "Polynomial Regression", "y = ax^2 + bx + c", None

def run_ridge_regression(X, y, alpha, metric):
    # Placeholder for running the model and generating results
    return None, 0.88, "Ridge Regression", "y = ax + b (Ridge)", None

def run_lasso_regression(X, y, alpha, metric):
    # Placeholder for running the model and generating results
    return None, 0.86, "Lasso Regression", "y = ax + b (Lasso)", None

def run_svr(X, y, metric):
    # Placeholder for running the model and generating results
    return None, 0.89, "Support Vector Regression", "SVR equation", None

def run_decision_tree_regression(X, y, metric):
    # Placeholder for running the model and generating results
    return None, 0.83, "Decision Tree Regression", "Decision Tree equation", None

def run_random_forest_regression(X, y, metric):
    # Placeholder for running the model and generating results
    return None, 0.91, "Random Forest Regression", "Random Forest equation", None

def main():
    def load_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            entry_file_path.delete(0, tk.END)
            entry_file_path.insert(0, file_path)
            with open(file_path, 'r') as file:
                text_csv.delete('1.0', tk.END)
                text_csv.insert(tk.END, file.read())

    def save_csv():
        try:
            file_path = entry_file_path.get()
            content = text_csv.get('1.0', tk.END)
            with open(file_path, 'w') as file:
                file.write(content)
            df = pd.read_csv(file_path)
            df.to_csv(file_path, index=False)
            label_status.config(text="CSV file saved successfully!", fg="green")
        except Exception as e:
            label_status.config(text=f"Error saving CSV file: {e}", fg="red")

    def run_models():
        csv_file = entry_file_path.get()
        X, y, data = load_data(csv_file)

        metric = combo_metric.get().lower()

        global TRAIN_DATA_COLOR, TEST_DATA_COLOR, REGRESSION_LINE_COLOR, VISUALIZATIONS_PER_ROW
        TRAIN_DATA_COLOR = entry_train_color.get()
        TEST_DATA_COLOR = entry_test_color.get()
        REGRESSION_LINE_COLOR = entry_regression_color.get()
        VISUALIZATIONS_PER_ROW = int(spinbox_visualizations_per_row.get())

        models = [
            run_simple_linear_regression(X, y, metric=metric),
            run_multiple_linear_regression(X, y, metric=metric),
            run_polynomial_regression(X.copy(), y, 2, metric=metric),
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
            label_result = tk.Label(scrollable_frame_summary, text=result)
            label_result.pack()

            img = tk.PhotoImage(data=plot_data)
            label_image = tk.Label(frame_visualizations, image=img)
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

        # Add heading for visualizations
        heading_visualizations = f"Visualizations for {csv_file} with {metric}"
        label_heading_visualizations = tk.Label(scrollable_frame_visualizations, text=heading_visualizations, fg="blue", font=("Helvetica", 16, "bold"))
        label_heading_visualizations.pack(side=tk.TOP, pady=10)

        # Arrange visualizations in a grid
        for i, child in enumerate(frame_visualizations.winfo_children()):
            child.grid(row=i // VISUALIZATIONS_PER_ROW, column=i % VISUALIZATIONS_PER_ROW, padx=5, pady=5)

    root = tk.Tk()
    root.title("Regression Model Comparison")

    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, fill=tk.BOTH)

    def add_scrollbars_to_tab(tab):
        canvas = tk.Canvas(tab)
        scrollbar_x = ttk.Scrollbar(tab, orient="horizontal", command=canvas.xview)
        scrollbar_y = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_x.pack(side="bottom", fill="x")
        scrollbar_y.pack(side="right", fill="y")

        return scrollable_frame

    # Tab 1: File selection and options
    tab_file = ttk.Frame(notebook)
    notebook.add(tab_file, text="File Selection and Options")
    scrollable_frame_file = add_scrollbars_to_tab(tab_file)

    frame_file = tk.Frame(scrollable_frame_file)
    frame_file.pack(pady=10)
    tk.Label(frame_file, text="CSV File:").pack(side=tk.LEFT)
    entry_file_path = tk.Entry(frame_file, width=50)
    entry_file_path.pack(side=tk.LEFT, padx=5)
    btn_browse = tk.Button(frame_file, text="Browse", command=load_file)
    btn_browse.pack(side=tk.LEFT)

    frame_options = tk.Frame(scrollable_frame_file)
    frame_options.pack(pady=10)
    tk.Label(frame_options, text="Accuracy Metric:").pack(side=tk.LEFT)
    combo_metric = ttk.Combobox(frame_options, values=["r2", "mse"], state="readonly")
    combo_metric.set("mse")
    combo_metric.pack(side=tk.LEFT, padx=5)

    # Additional options for visualization settings
    frame_visual_options = tk.Frame(scrollable_frame_file)
    frame_visual_options.pack(pady=10)
    
    tk.Label(frame_visual_options, text="Train Data Color:").grid(row=0, column=0, padx=5, pady=5)
    entry_train_color = tk.Entry(frame_visual_options)
    entry_train_color.insert(0, TRAIN_DATA_COLOR)
    entry_train_color.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(frame_visual_options, text="Test Data Color:").grid(row=1, column=0, padx=5, pady=5)
    entry_test_color = tk.Entry(frame_visual_options)
    entry_test_color.insert(0, TEST_DATA_COLOR)
    entry_test_color.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(frame_visual_options, text="Regression Line Color:").grid(row=2, column=0, padx=5, pady=5)
    entry_regression_color = tk.Entry(frame_visual_options)
    entry_regression_color.insert(0, REGRESSION_LINE_COLOR)
    entry_regression_color.grid(row=2, column=1, padx=5, pady=5)

    tk.Label(frame_visual_options, text="Visualizations per Row:").grid(row=3, column=0, padx=5, pady=5)
    spinbox_visualizations_per_row = tk.Spinbox(frame_visual_options, from_=1, to=10, width=5)
    spinbox_visualizations_per_row.delete(0, tk.END)
    spinbox_visualizations_per_row.insert(0, VISUALIZATIONS_PER_ROW)
    spinbox_visualizations_per_row.grid(row=3, column=1, padx=5, pady=5)

    # Tab 2: Edit CSV
    tab_edit_csv = ttk.Frame(notebook)
    notebook.add(tab_edit_csv, text="Edit CSV")
    scrollable_frame_edit_csv = add_scrollbars_to_tab(tab_edit_csv)

    frame_edit_csv = tk.Frame(scrollable_frame_edit_csv)
    frame_edit_csv.pack(pady=10)
    text_csv = ScrolledText(frame_edit_csv, width=80, height=20)
    text_csv.pack(padx=10, pady=10)

    frame_save = tk.Frame(scrollable_frame_edit_csv)
    frame_save.pack(pady=10)
    btn_save = tk.Button(frame_save, text="Save CSV", command=save_csv)
    btn_save.pack(side=tk.LEFT, padx=5)
    label_status = tk.Label(frame_save, text="", fg="red")
    label_status.pack(side=tk.LEFT, padx=5)

    # Tab 3: Summary
    tab_summary = ttk.Frame(notebook)
    notebook.add(tab_summary, text="Summary")
    scrollable_frame_summary = add_scrollbars_to_tab(tab_summary)

    # Tab 4: Visualizations
    tab_visualizations = ttk.Frame(notebook)
    notebook.add(tab_visualizations, text="Visualizations")
    scrollable_frame_visualizations = add_scrollbars_to_tab(tab_visualizations)
    frame_visualizations = tk.Frame(scrollable_frame_visualizations)
    frame_visualizations.pack(pady=10)

    # Run Models Button
    frame_run_models = tk.Frame(scrollable_frame_file)
    frame_run_models.pack(pady=10)
    btn_run_models = tk.Button(frame_run_models, text="Run Models", command=run_models)
    btn_run_models.pack(side=tk.LEFT, padx=5)
    label_best_model = tk.Label(frame_run_models, text="", fg="green")
    label_best_model.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()
