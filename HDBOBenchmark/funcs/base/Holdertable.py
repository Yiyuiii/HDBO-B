from .FunctionBase import BOTorchTestFunction
from botorch.test_functions import HolderTable as HolderTable_


class HolderTable(BOTorchTestFunction):
    def __init__(
        self,
        dim,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(HolderTable_(), "HolderTable")
        self.y_scale = 20
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
