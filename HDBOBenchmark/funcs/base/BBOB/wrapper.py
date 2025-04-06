import numpy as np
from ..FunctionBase import TestFunction


class VizierBBOBWrapper(TestFunction):
    def __init__(
        self,
        func,
        dim,
        seed: int = 0,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=False,
    ):
        super().__init__(dim, func.__name__)

        self.func = func
        self.seed = seed
        self._x_bound = np.array([[-5, 5]] * self.dim)  # shape(dim, 2)
        self.y_scale = 1
        self._min_pt = np.zeros(self.dim)
        # self._min_val = self._evaluate(self._min_pt)
        self._min_val = 0

        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def _evaluate(
        self, x: np.ndarray
    ) -> float:  # black-box objective function to minimize
        assert x.ndim <= 2
        if x.ndim == 2:  # batch integration
            y = list()
            for x_ in x:
                y.append(self.func(x_, self.seed))
            return np.array(y)
        return self.func(x, self.seed)

    def to_torch(self, device="cpu"):
        raise NotImplementedError("BBOB is implemented with Numpy.")

    def set_seed(self, seed: int):
        self.seed = seed
