import numpy as np
import torch
from typing import Optional, Union
from .FunctionBase import TestFunction


class AbsMax(TestFunction):
    def __init__(
        self,
        dim,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(dim, "AbsMax")

        self._x_bound = np.array([[-1, 1]] * self.dim, dtype=float)  # shape(dim, 2)
        self.y_scale = 1
        self._min_pt = np.zeros(self.dim)
        # self._min_val = self._evaluate(self._min_pt)
        self._min_val = 0

        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def _evaluate(
        self, x: Union[torch.Tensor, np.ndarray]
    ):  # black-box objective function to minimize
        if self.if_torch:
            y = x.abs().max(dim=-1).values
        else:
            y = np.abs(x).max(axis=-1)
        return y
