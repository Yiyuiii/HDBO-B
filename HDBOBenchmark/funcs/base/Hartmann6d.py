from .FunctionBase import BOTorchTestFunction
from botorch.test_functions import Hartmann as Hartmann_


class Hartmann6d(BOTorchTestFunction):
    def __init__(
        self,
        dim,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(Hartmann_(dim=6), "Hartmann6d")
        self.y_scale = 3
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
