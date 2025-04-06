from .FunctionBase import BOTorchTestFunction
from botorch.test_functions import DropWave as DropWave_


class DropWave(BOTorchTestFunction):
    def __init__(
        self,
        dim,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(DropWave_(), "DropWave")
        self.y_scale = 1
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
