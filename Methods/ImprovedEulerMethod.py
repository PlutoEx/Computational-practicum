import numpy as np


class ImprovedEulerMethod:
    def __init__(self, h, y_prime_formula):
        self.h = h
        self.y_prime_formula = y_prime_formula

    def get_y_prime(self, x, y):
        return eval(self.y_prime_formula, {"x": x, "y": y, "np": np})

    def get_y(self, x_prev, y_prev, h):
        k1 = self.get_y_prime(x_prev, y_prev)
        k2 = self.get_y_prime(x_prev + h, y_prev + h * k1)
        return y_prev + h * (k1 + k2) / 2
