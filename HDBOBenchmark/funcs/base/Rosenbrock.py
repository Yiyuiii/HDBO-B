from .FunctionBase import BOTorchTestFunction
from botorch.test_functions import Rosenbrock as Rosenbrock_


class Rosenbrock(BOTorchTestFunction):
    def __init__(
        self,
        dim,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
    ):
        super().__init__(
            Rosenbrock_(dim=dim, bounds=[(-2.048, 2.048) for _ in range(dim)]),
            "Rosenbrock",
        )
        self.y_scale = 2000 * self.dim
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
