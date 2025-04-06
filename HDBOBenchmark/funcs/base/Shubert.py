import numpy as np
from .FunctionBase import TestFunction


class Shubert(TestFunction):
    def __init__(
        self,
        dim,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(2, "Shubert")

        self._x_bound = np.array([[-10, 10]] * self.dim, dtype=float)  # shape(dim, 2)
        self.y_scale = 17 * self.dim
        self._min_pt = None  # ** can not find the exact min points **
        self._min_val = -16

        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def _evaluate(self, x):  # black-box objective function to minimize
        sum = 0
        for j in range(self.dim):
            xj = x[j]
            for i in range(5):
                sum += i * np.cos((i + 1) * xj + i)
        y = sum
        return y
