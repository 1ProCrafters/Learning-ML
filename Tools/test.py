# Define the equation with terms in any order
equation = "0.5*x^2 - 3*x + 7"

# Rearrange the terms from the smallest degree to the highest degree
terms = equation.split(" ")
terms.sort(key=lambda x: x.count("x^"))

# Construct the equation in the form y hat
predicted_equation = "y_hat = " + " + ".join(terms)

print(predicted_equation)