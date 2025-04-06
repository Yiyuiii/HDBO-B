import numpy as np
import torch
from .FunctionBase import TestFunction


class Sin(TestFunction):
    def __init__(
        self,
        dim,
        a=1,
        b=1,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(dim, "Sin")

        self.a, self.b = a, b
        self._x_bound = np.array(
            [[-np.pi * 4, np.pi * 4]] * self.dim, dtype=float
        )  # shape(dim, 2)
        self._min_pt = -torch.pi * torch.ones(self.dim) / 2
        self._min_val = -dim
        self.y_scale = 2 * dim
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
        self.to_torch()

    def _evaluate(self, x):  # black-box objective function to minimize
        return torch.sin(self.a * x + self.b).sum(dim=-1)
