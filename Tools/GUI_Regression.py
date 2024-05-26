# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import PolynomialFeatures
# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk, Text

# TRAIN_DATA_COLOR = 'blue'
# TEST_DATA_COLOR = 'red'
# REGRESSION_LINE_COLOR = 'green'

# class RegressionApp:
#     def __init__(self, master):
#         self.master = master
#         self.master.title("Regression Model Visualizer")

#         self.notebook = ttk.Notebook(master)
#         self.notebook.pack(expand=1, fill="both")

#         self.create_options_tab()
#         self.create_csv_viewer_tab()
#         self.create_regressions_tab()

#     def create_options_tab(self):
#         self.options_frame = ttk.Frame(self.notebook)
#         self.notebook.add(self.options_frame, text="Options")

#         self.label = ttk.Label(self.options_frame, text="Choose a CSV file")
#         self.label.pack(pady=10)

#         self.filepath = tk.StringVar()
#         self.file_entry = ttk.Entry(self.options_frame, textvariable=self.filepath, width=50)
#         self.file_entry.pack(pady=5)

#         self.browse_button = ttk.Button(self.options_frame, text="Browse", command=self.browse_file)
#         self.browse_button.pack(pady=5)

#         self.load_button = ttk.Button(self.options_frame, text="Load CSV", command=self.load_csv)
#         self.load_button.pack(pady=5)

#         self.run_button = ttk.Button(self.options_frame, text="Run Regressions", command=self.run_regressions)
#         self.run_button.pack(pady=20)

#         self.visualize = tk.BooleanVar()
#         self.visualize.set(True)
#         self.visualize_check = ttk.Checkbutton(self.options_frame, text="Visualize Data", variable=self.visualize)
#         self.visualize_check.pack()

#         self.metric_label = ttk.Label(self.options_frame, text="Metric (r2 or mse)")
#         self.metric_label.pack(pady=5)

#         self.metric_entry = ttk.Entry(self.options_frame)
#         self.metric_entry.pack(pady=5)

#         self.advanced_details_button = ttk.Button(self.options_frame, text="Advanced Details", command=self.show_advanced_details)
#         self.advanced_details_button.pack(pady=10)

#     def create_csv_viewer_tab(self):
#         self.csv_viewer_frame = ttk.Frame(self.notebook)
#         self.notebook.add(self.csv_viewer_frame, text="CSV Viewer")

#         self.text_widget = Text(self.csv_viewer_frame, wrap="none")
#         self.text_widget.pack(expand=1, fill="both")

#         self.scrollbar_y = ttk.Scrollbar(self.csv_viewer_frame, orient="vertical", command=self.text_widget.yview)
#         self.scrollbar_y.pack(side="right", fill="y")
#         self.text_widget.config(yscrollcommand=self.scrollbar_y.set)

#         self.scrollbar_x = ttk.Scrollbar(self.csv_viewer_frame, orient="horizontal", command=self.text_widget.xview)
#         self.scrollbar_x.pack(side="bottom", fill="x")
#         self.text_widget.config(xscrollcommand=self.scrollbar_x.set)

#     def create_regressions_tab(self):
#         self.regressions_frame = ttk.Frame(self.notebook)
#         self.notebook.add(self.regressions_frame, text="Regressions")

#         self.regression_text = Text(self.regressions_frame, wrap="word")
#         self.regression_text.pack(expand=1, fill="both")

#         self.regression_scrollbar_y = ttk.Scrollbar(self.regressions_frame, orient="vertical", command=self.regression_text.yview)
#         self.regression_scrollbar_y.pack(side="right", fill="y")
#         self.regression_text.config(yscrollcommand=self.regression_scrollbar_y.set)

#     def browse_file(self):
#         self.filepath.set(filedialog.askopenfilename())

#     def load_csv(self):
#         try:
#             with open(self.filepath.get(), "r") as file:
#                 csv_content = file.read()
#             self.text_widget.delete(1.0, tk.END)
#             self.text_widget.insert(tk.END, csv_content)
#         except FileNotFoundError:
#             messagebox.showerror("Error", "CSV file not found. Please try again.")
#         except Exception as e:
#             messagebox.showerror("Error", f"An error occurred: {str(e)}")

#     def show_advanced_details(self):
#         advanced_window = tk.Toplevel(self.master)
#         advanced_window.title("Advanced Details")

#         advanced_text = Text(advanced_window, wrap="word")
#         advanced_text.pack(expand=True, fill="both")
#         advanced_text.insert("1.0", self.summary)

#     def load_data(self, csv_file):
#         try:
#             data = pd.read_csv(csv_file)
#             X = data.drop('target', axis=1)
#             y = data['target']
#             return X, y
#         except FileNotFoundError:
#             messagebox.showerror("Error", "CSV file not found. Please try again.")
#             return None, None
#         except pd.errors.EmptyDataError:
#             messagebox.showerror("Error", "The selected CSV file is empty.")
#             return None, None
#         except pd.errors.ParserError:
#             messagebox.showerror("Error", "Error parsing CSV file. Please check the file format.")
#             return None, None
#         except KeyError:
#             messagebox.showerror("Error", "CSV file must contain a 'target' column.")
#             return None, None

#     def generate_simplified_equation(self, model, feature_names=None, threshold=1e-4):
#         if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
#             coef = model.coef_
#             intercept = model.intercept_
#             terms = []
#             if feature_names is None:
#                 feature_names = [f"x{i}" for i in range(len(coef))]
#             for i, c in enumerate(coef):
#                 if abs(c) > threshold:
#                     terms.append(f"{c:.4f}*{feature_names[i]}")
#             equation = " + ".join(terms)
#             if abs(intercept) > threshold:
#                 equation = f"{intercept:.4f} + " + equation
#             return equation
#         return "Equation not available for this model."

#     def run_regression_model(self, X, y, model_name, model_func, metric_func, visualize_data, feature_names=None):
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         model = model_func()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         accuracy = metric_func(y_test, y_pred)
#         equation = self.generate_simplified_equation(model, feature_names)

#         if visualize_data:
#             plt.figure()
#             plt.scatter(X_train.iloc[:, 0], y_train, color=TRAIN_DATA_COLOR, label='Train Data')
#             plt.scatter(X_test.iloc[:, 0], y_test, color=TEST_DATA_COLOR, label='Test Data')
#             plt.plot(X_test.iloc[:, 0], y_pred, color=REGRESSION_LINE_COLOR, label='Regression Line')
#             plt.xlabel('Feature')
#             plt.ylabel('Target Variable')
#             plt.title(f"{model_name} Visualization")
#             plt.legend()
#             plot_name = f"{model_name.replace(' ', '_')}_visualization.png"
#             plt.savefig(plot_name)
#             plt.close()
#             self.show_plot(plot_name, model_name)

#         return model, accuracy, model_name, equation

#     def show_plot(self, plot_name, title):
#         plot_window = tk.Toplevel(self.master)
#         plot_window.title(title)

#         img = plt.imread(plot_name)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(title)
#         plt.show()

#     def run_simple_linear_regression(self, X, y, visualize_data, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         feature_names = [X.columns[0]]
#         X_feature = X.iloc[:, 0].values.reshape(-1, 1)
#         return self.run_regression_model(pd.DataFrame(X_feature, columns=feature_names), y, "Simple Linear Regression", LinearRegression, metric_func, visualize_data, feature_names)

#     def run_multiple_linear_regression(self, X, y, visualize_data, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         feature_names = X.columns
#         return self.run_regression_model(X, y, "Multiple Linear Regression", LinearRegression, metric_func, visualize_data, feature_names)

#     def run_polynomial_regression(self, X, y, visualize_data, degree, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         polynomial_features = PolynomialFeatures(degree=degree)
#         X_poly = polynomial_features.fit_transform(X)
#         model = LinearRegression()
#         model.fit(X_poly, y)
#         y_pred = model.predict(X_poly)

#         feature_names = polynomial_features.get_feature_names_out(X.columns)
#         equation = self.generate_simplified_equation(model, feature_names)

#         if visualize_data:
#             plt.figure()
#             plt.scatter(X.iloc[:, 0], y, color=TRAIN_DATA_COLOR, label='Data')
#             plt.plot(X.iloc[:, 0], y_pred, color=REGRESSION_LINE_COLOR, label='Polynomial Regression Line')
#             plt.xlabel('Feature')
#             plt.ylabel('Target Variable')
#             plt.title(f"Polynomial Regression (Degree: {degree}) Visualization")
#             plt.legend()
#             plot_name = f"Polynomial_Regression_Degree_{degree}_visualization.png"
#             plt.savefig(plot_name)
#             plt.close()
#             self.show_plot(plot_name, f"Polynomial Regression (Degree: {degree})")

#         accuracy = metric_func(y, y_pred)
#         model_name = f"Polynomial Regression (Degree: {degree})"
#         return model, accuracy, model_name, equation

#     def run_ridge_regression(self, X, y, visualize_data, alpha=1.0, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         feature_names = X.columns
#         return self.run_regression_model(X, y, "Ridge Regression", lambda: Ridge(alpha=alpha), metric_func, visualize_data, feature_names)

#     def run_lasso_regression(self, X, y, visualize_data, alpha=0.1, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         feature_names = X.columns
#         model, accuracy, model_name, equation = self.run_regression_model(X, y, "Lasso Regression", lambda: Lasso(alpha=alpha), metric_func, visualize_data, feature_names)

#         if visualize_data:
#             plt.figure()
#             plt.scatter(X.iloc[:, 0], y, color=TRAIN_DATA_COLOR, label='Data')
#             plt.plot(X.iloc[:, 0], model.predict(X), color=REGRESSION_LINE_COLOR, label='Regression Line')
#             plt.xlabel('Feature')
#             plt.ylabel('Target Variable')
#             plt.title(f"Lasso Regression (Alpha: {alpha}) Visualization")
#             plt.legend()
#             plot_name = f"Lasso_Regression_Alpha_{alpha}_visualization.png"
#             plt.savefig(plot_name)
#             plt.close()
#             self.show_plot(plot_name, f"Lasso Regression (Alpha: {alpha})")

#         return model, accuracy, model_name, equation

#     def run_svr(self, X, y, visualize_data, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         return self.run_regression_model(X, y, "Support Vector Regression (SVR)", SVR, metric_func, visualize_data)

#     def run_decision_tree_regression(self, X, y, visualize_data, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         return self.run_regression_model(X, y, "Decision Tree Regression", DecisionTreeRegressor, metric_func, visualize_data)

#     def run_random_forest_regression(self, X, y, visualize_data, metric="r2"):
#         metric_func = r2_score if metric == "r2" else mean_squared_error
#         return self.run_regression_model(X, y, "Random Forest Regression", RandomForestRegressor, metric_func, visualize_data)

#     def run_regressions(self):
#         csv_file = self.filepath.get()
#         X, y = self.load_data(csv_file)

#         if X is None or y is None:
#             return

#         metric = self.metric_entry.get().strip().lower() or "mse"
#         visualize_data = self.visualize.get()

#         models = [
#             self.run_simple_linear_regression(X, y, visualize_data, metric=metric),
#             self.run_multiple_linear_regression(X, y, visualize_data, metric=metric),
#             self.run_polynomial_regression(X.copy(), y, visualize_data, degree=2, metric=metric),
#             self.run_ridge_regression(X, y, visualize_data, alpha=1.0, metric=metric),
#             self.run_lasso_regression(X, y, visualize_data, alpha=0.1, metric=metric),
#             self.run_svr(X, y, visualize_data, metric=metric),
#             self.run_decision_tree_regression(X, y, visualize_data, metric=metric),
#             self.run_random_forest_regression(X, y, visualize_data, metric=metric),
#         ]

#         best_model = None
#         best_accuracy = None
#         best_equation = None
#         best_model_name = None
        
#         condition = "greater_than" if metric == "r2" else "less_than"
#         operators = {
#             "greater_than": ">",
#             "less_than": "<",
#             "equal_to": "=="
#         }

#         self.summary = ""
#         for model, accuracy, model_name, equation in models:
#             if not equation:
#                 equation = "Equation not available for this model."
#             else:
#                 equation = "Equation: " + equation
#             self.summary += f"{model_name}: Accuracy = {accuracy:.4f}; {equation}\n"
            
#             if best_accuracy is None:
#                 best_accuracy = accuracy
#             if eval(str(accuracy) + str(operators[condition]) + str(best_accuracy)):
#                 best_model = model
#                 best_accuracy = accuracy
#                 best_equation = equation
#                 best_model_name = model_name

#         if best_model:
#             self.summary += f"\nBest Model: '{best_model_name}' with Accuracy = {best_accuracy:.4f} and {best_equation}"
#         else:
#             self.summary += "\nNo models were run successfully."

#         self.regression_text.delete(1.0, tk.END)
#         self.regression_text.insert(tk.END, self.summary)

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = RegressionApp(root)
#     root.mainloop()
