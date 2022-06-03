import numpy as np


class EulerMethod:
    def __init__(self, h, y_prime_formula):
        self.h = h
        self.y_prime_formula = y_prime_formula

    def get_y_prime(self, x, y):
        return eval(self.y_prime_formula, {"x": x, "y": y, "np": np})

    def get_y(self, x_prev, y_prev, h):
        return y_prev + h * self.get_y_prime(x_prev, y_prev)
