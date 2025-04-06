import numpy as np
from .FunctionBase import TestFunction


class Trid(TestFunction):
    def __init__(
        self,
        dim,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(dim, "Trid")

        self._x_bound = np.array(
            [[-self.dim * self.dim, self.dim * self.dim]] * self.dim, dtype=float
        )  # shape(dim, 2)
        self.y_scale = 6 * self.dim**3
        self._min_pt = np.zeros(self.dim)
        for i in range(self.dim):
            self._min_pt[i] = (i + 1) * (self.dim - i)
        # self._min_val = self._evaluate(self._min_pt)
        self._min_val = -self.dim * (self.dim + 4) * (self.dim - 1) / 6

        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def _evaluate(self, x):  # black-box objective function to minimize
        xi_ = x[..., 0]
        sum = (xi_ - 1) ** 2
        for i in range(1, self.dim):
            xi = x[..., i]
            sum += (xi - 1) ** 2 - xi * xi_
            xi_ = xi
        y = sum
        return y
