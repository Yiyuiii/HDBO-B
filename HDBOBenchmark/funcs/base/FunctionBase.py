from typing import Optional
import numpy as np
import torch
import abc


class TestFunction(object):
    def __init__(self, dim=1, name="TestFunction", x_scaler=None, y_scaler=None):
        self._name = name
        self._dim = dim
        self._x_bound = np.array([[-1, 1]] * self.dim, dtype=float)  # shape(dim, 2)
        self._min_pt = None
        self._min_val = None
        self.y_scale = 100  # For function auto_init_y_scaler

        self._if_torch = False
        self.if_minium = True

    @staticmethod
    def _cal_scaled(a, a_scaler):
        if isinstance(a, np.ndarray) and len(a.shape) == 2:
            return (a + np.tile(a_scaler[0], (a.shape[1], 1)).transpose()) * np.tile(
                a_scaler[1], (a.shape[1], 1)
            ).transpose()
        elif isinstance(a, torch.Tensor) and len(a.shape) == 2:
            return (a + torch.tile(a_scaler[0], (a.shape[1], 1)).T) * torch.tile(
                a_scaler[1], (a.shape[1], 1)
            ).T
        return (a + a_scaler[0]) * a_scaler[1]

    @staticmethod
    def _cal_descaled(a_scaled, a_scaler):
        if isinstance(a_scaled, np.ndarray) and len(a_scaled.shape) == 2:
            return (a_scaled / np.tile(a_scaler[1], (a_scaled.shape[0], 1))) - np.tile(
                a_scaler[0], (a_scaled.shape[0], 1)
            )
        elif isinstance(a_scaled, torch.Tensor) and len(a_scaled.shape) == 2:
            return (
                a_scaled / torch.tile(a_scaler[1], (a_scaled.shape[0], 1))
            ) - torch.tile(a_scaler[0], (a_scaled.shape[0], 1))
        return (a_scaled / a_scaler[1]) - a_scaler[0]

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bound(self) -> np.ndarray:
        return self._cal_scaled(self._x_bound, self.x_scaler)

    @property
    def min_pt(self) -> Optional[float]:
        if self._min_pt is None:
            return None
        return self._cal_scaled(self._min_pt, self.x_scaler)

    @property
    def min_val(self) -> Optional[float]:
        if self._min_val is None:
            return None
        return self._cal_scaled(self._min_val, self.y_scaler)

    @property
    def if_torch(self) -> bool:
        return self._if_torch

    @abc.abstractmethod
    def _evaluate(self, x) -> Optional[float]:
        return

    def _x_preprocess(self, x):
        """
        Return x: (batch, feature)
        """
        if self._if_torch:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).to(self.x_scaler)
            while x.dim() < 2:
                x = x.unsqueeze(0)
        else:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            elif not isinstance(x, np.ndarray):
                x = np.array(x)
            while x.ndim < 2:
                x = x[np.newaxis]
        return x

    def evaluate(self, x) -> dict:
        x = self._x_preprocess(x)
        result = dict()
        result["x"] = x
        y = self(x)
        result["y"] = y
        return result

    def __call__(self, x) -> float:
        x = self._x_preprocess(x)
        x = self._cal_descaled(x, self.x_scaler)
        y = self._evaluate(x)
        y = self._cal_scaled(y, self.y_scaler)
        if not self.if_minium:
            y = -y
        return y

    def init_scaler(
        self,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        self.x_scaler = (
            np.array([np.zeros(self.dim), np.ones(self.dim)], dtype=float)
            if x_scaler is None
            else x_scaler
        )
        self.y_scaler = np.array([0, 1], dtype=float) if y_scaler is None else y_scaler
        if auto_init_x_scaler:
            self.auto_init_x_scaler()
        if auto_init_y_scaler:
            self.auto_init_y_scaler()

    def auto_init_x_scaler(self):
        self.x_scaler[0] = (self._x_bound[:, 1] + self._x_bound[:, 0]) * -0.5
        self.x_scaler[1] = 2 / (self._x_bound[:, 1] - self._x_bound[:, 0])

    def auto_init_y_scaler(self):
        self.y_scaler[0] = -self._min_val
        self.y_scaler[1] = 100 / self.y_scale

    # only if function is in torch
    def to_torch(self, device="cpu"):
        if not self._if_torch:
            self._if_torch = True
            self._min_pt = torch.tensor(self._min_pt, device=device)
            self._x_bound = torch.tensor(self._x_bound, device=device)
            self.x_scaler = torch.tensor(self.x_scaler, device=device)
            self.y_scaler = torch.tensor(self.y_scaler, device=device)

    def switch_maximum(self):
        self.if_minium = False

    def switch_minium(self):
        self.if_minium = True


class BOTorchTestFunction(TestFunction):
    def __init__(
        self,
        func,
        name="BOTorchTestFunction",
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        self.botorch_func = func
        super().__init__(self.botorch_func.dim, name)
        self._x_bound = np.array(self.botorch_func._bounds)  # shape(dim, 2)
        self._min_pt = np.array(self.botorch_func._optimizers[0])
        # self._min_val = self._evaluate(self._min_pt)
        self._min_val = self.botorch_func.optimal_value
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
        self.to_torch()

    def _evaluate(self, x):  # black-box objective function to minimize
        return self.botorch_func(x)

    def init_scaler(
        self, x_scaler, y_scaler, auto_init_x_scaler=True, auto_init_y_scaler=True
    ):
        super().init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
        if not isinstance(self.x_scaler, torch.Tensor):
            self.x_scaler = torch.tensor(self.x_scaler)
