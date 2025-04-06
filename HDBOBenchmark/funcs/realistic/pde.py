"""
==================================================
Repo: https://github.com/zwicker-group/py-pde
Doc: https://py-pde.readthedocs.io/en/latest/index.html
Install by:
pip install py-pde
==================================================
"""

import numpy as np
from HDBOBenchmark.funcs.base.FunctionBase import TestFunction
from pde import PDE, FieldCollection, ScalarField, UnitGrid


class Brusselator(TestFunction):
    """
    https://py-pde.readthedocs.io/en/latest/examples_gallery/pde_brusselator_expression.html#sphx-glr-examples-gallery-pde-brusselator-expression-py
    References: https://github.com/bhouri0412/rpn_bo
    """

    def __init__(
        self,
        dim=4,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=False,
    ):
        super().__init__(4, "Brusselator")
        self._grid = UnitGrid([64, 64])

        self._x_bound = np.array(
            [[0.1, 5], [0.1, 5], [0.01, 5], [0.01, 5]], dtype=float
        )  # shape(dim, 2)
        self.y_scale = 3
        self._min_pt = None
        self._min_val = 0
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def _evaluate(self, x):  # black-box objective function to minimize
        a, b, d0, d1 = x[0], x[1], x[2], x[3]
        eq = PDE(
            {
                "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
            }
        )
        u = ScalarField(self._grid, a, label="Field $u$")
        v = b / a + 0.1 * ScalarField.random_normal(self._grid, label="Field $v$")
        state = FieldCollection([u, v])
        sol = eq.solve(state, t_range=20, dt=1e-3)
        y = sol.data.mean(
            axis=0
        ).max()  # maximum average density of u,v in a place at the last time t
        del eq, u, v, state, sol
        return y
