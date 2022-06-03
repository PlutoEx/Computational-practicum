import numpy as np
import matplotlib.pyplot as plt
from Methods.EulerMethod import EulerMethod
from Methods.ImprovedEulerMethod import ImprovedEulerMethod
from Methods.RungeKuttaMethod import RungeKuttaMethod


class Gui:
    def __init__(self, x_start, y0, x_end, N, y_prime_formula, y_formula):
        self.x_start = x_start
        self.y0 = y0
        self.x_end = x_end
        self.N = N
        self.y_prime_formula = y_prime_formula
        self.C = self.y0 / self.x_start - np.sin(x_start)
        self.y_formula = y_formula

        h = (self.x_end - self.x_start) / (self.N - 1)

        # Creating arrays
        self.x = np.arange(self.x_start, self.x_end + h / 2, h)
        self.y_exact = self.y_euler = self.y_imp_euler = self.y_runge_kutta = np.array([y0])
        self.lte_euler = self.lte_imp_euler = self.lte_runge_kutta = np.array([0])
        self.gte_euler = self.gte_imp_euler = self.gte_runge_kutta = np.array([0])
        self.sum_euler = self.sum_imp_euler = self.sum_runge_kutta = np.array([0])

        self.euler_method = EulerMethod(h, y_prime_formula)
        self.imp_euler_method = ImprovedEulerMethod(h, y_prime_formula)
        self.runge_kutta_method = RungeKuttaMethod(h, y_prime_formula)

    def solve(self):
        # Exact, Euler, Improved Euler, Runge-Kutta methods, LTE, GTE
        h = (self.x_end - self.x_start) / (self.N - 1)
        for i in range(1, self.N, 1):
            self.y_exact = np.append(self.y_exact, eval(self.y_formula, {"x": self.x[i], "np": np, "C": self.C}))
            self.y_euler = np.append(self.y_euler, self.euler_method.get_y(self.x[i - 1], self.y_euler[i - 1], h))
            self.y_imp_euler = np.append(self.y_imp_euler, self.imp_euler_method.get_y(self.x[i - 1], self.y_imp_euler[i - 1], h))
            self.y_runge_kutta = np.append(self.y_runge_kutta, self.runge_kutta_method.get_y(self.x[i - 1], self.y_runge_kutta[i - 1], h))
            self.lte_euler = np.append(self.lte_euler,
                abs(self.euler_method.get_y(self.x[i - 1], self.y_exact[i - 1], h) - self.y_exact[i]))
            self.lte_imp_euler = np.append(self.lte_imp_euler,
                abs(self.imp_euler_method.get_y(self.x[i - 1], self.y_exact[i - 1], h) - self.y_exact[i]))
            self.lte_runge_kutta = np.append(self.lte_runge_kutta,
                abs(self.runge_kutta_method.get_y(self.x[i - 1], self.y_exact[i - 1], h) - self.y_exact[i]))
            self.gte_euler = np.append(self.gte_euler, abs(self.y_exact[i] - self.y_euler[i]))
            self.gte_imp_euler = np.append(self.gte_imp_euler, abs(self.y_exact[i] - self.y_imp_euler[i]))
            self.gte_runge_kutta = np.append(self.gte_runge_kutta, abs(self.y_exact[i] - self.y_runge_kutta[i]))

        # Errors depending on number of points on graph (N)
        for n in range(2, 100, 1):
            h = (self.x_end - self.x_start) / (n - 1)
            x2 = np.arange(self.x_start, self.x_end + h / 2, h)
            temp_euler = temp_imp_euler = temp_runge_kutta = 0
            exact = euler = imp_euler = runge_kutta = self.y0
            for i in range(1, n, 1):
                exact = eval(self.y_formula, {"x": x2[i], "np": np, "C": self.C})
                euler = self.euler_method.get_y(x2[i - 1], euler, h)
                imp_euler = self.imp_euler_method.get_y(x2[i - 1], imp_euler, h)
                runge_kutta = self.runge_kutta_method.get_y(x2[i - 1], runge_kutta, h)
                temp_euler += abs(exact - euler)
                temp_imp_euler += abs(exact - imp_euler)
                temp_runge_kutta += abs(exact - runge_kutta)
            self.sum_euler = np.append(self.sum_euler, temp_euler)
            self.sum_imp_euler = np.append(self.sum_imp_euler, temp_imp_euler)
            self.sum_runge_kutta = np.append(self.sum_runge_kutta, temp_runge_kutta)

    def show(self):
        # Plotting the points
        plt.figure(1)
        plt.title("Exact & Numerical")
        plt.plot(self.x, self.y_exact, label="Exact", linestyle='-')
        plt.plot(self.x, self.y_euler, label="Euler", linestyle='--')
        plt.plot(self.x, self.y_imp_euler, label="Improved Euler", linestyle='-.')
        plt.plot(self.x, self.y_runge_kutta, label="Runge-Kutta", linestyle=':')
        plt.legend()
        plt.grid(True)
        plt.savefig("Reports/ExactAndNumerical.png")

        plt.figure(2)
        plt.title("LTE")
        plt.plot(self.x, self.lte_euler, label="Euler")
        plt.plot(self.x, self.lte_imp_euler, label="Improved Euler")
        plt.plot(self.x, self.lte_runge_kutta, label="Runge-Kutta")
        plt.legend()
        plt.grid(True)
        plt.savefig("Reports/LTE.png")

        plt.figure(3)
        plt.title("GTE")
        plt.plot(self.x, self.gte_euler, label="Euler")
        plt.plot(self.x, self.gte_imp_euler, label="Improved Euler")
        plt.plot(self.x, self.gte_runge_kutta, label="Runge-Kutta")
        plt.legend()
        plt.grid(True)
        plt.savefig("Reports/GTE.png")

        plt.figure(4)
        plt.title("Sum of errors depending on number of points on graph")
        plt.plot(self.sum_euler, label="Euler")
        plt.plot(self.sum_imp_euler, label="Improved Euler")
        plt.plot(self.sum_runge_kutta, label="Runge-Kutta")
        plt.legend()
        plt.grid(True)
        plt.savefig("Reports/SumOfErrors.png")

        plt.show()
