from Methods.Gui import Gui
import numpy as np

# Default
x_start = np.pi
x_end = 4 * np.pi
N = 1000
y0 = 1
y_prime_formula = "y / x + x * np.cos(x)"
y_formula = "x * (np.sin(x) + C)"

# Inputs
print("Default: pi, 1, 4*pi, 1000      //use np.pi to enter pi")
x_start = eval(input("Enter x0: "))
y0 = eval(input("Enter y0: "))
x_end = eval(input("Enter X: "))
N = eval(input("Enter N: "))

gui = Gui(x_start, y0, x_end, N, y_prime_formula, y_formula)
gui.solve()
gui.show()