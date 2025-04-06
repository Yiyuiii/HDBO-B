"""
==================================================
Repo: https://github.com/ksehic/LassoBench
Install by:
git clone https://github.com/ksehic/LassoBench.git
cd LassoBench/
pip install -e .
==================================================
"""

import numpy as np
from HDBOBenchmark.funcs.base.FunctionBase import TestFunction
import LassoBench


class LassoBenchmark(TestFunction):
    def __init__(
        self,
        benchname="synt_simple",
        dim=None,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=False,
    ):
        if "synt_" in benchname:
            self.lassobenchfunc = LassoBench.SyntheticBenchmark(pick_bench=benchname)
        else:
            self.lassobenchfunc = LassoBench.RealBenchmark(pick_data=benchname)

        super().__init__(dim=self.lassobenchfunc.n_features, name=benchname)

        self._x_bound = np.array([[-1, 1]] * self.dim, dtype=float)  # shape(dim, 2)
        y_scale_dict = {
            "Breast_cancer": 6.29,
            "Diabetes": 0.90,
            "DNA": 0.59,
            "leukemia": 7.73,
            "RCV": 0.32,
        }
        min_val_dict = {
            "Breast_cancer": 0.2609,
            "Diabetes": 0.648,
            "DNA": 0.292,
            "leukemia": 0.015,
            "RCV": 0.18,
        }
        self.y_scale = (
            y_scale_dict[benchname] if benchname in y_scale_dict.keys() else 100
        )
        self._min_pt = None
        self._min_val = (
            min_val_dict[benchname] if benchname in min_val_dict.keys() else 0
        )
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def _evaluate(self, x):  # black-box objective function to minimize
        x = np.clip(x, -1, 1)
        if x.ndim == 1:
            return self.lassobenchfunc.evaluate(x)
        else:
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            result = list()
            for x_slice in x:
                result.append(self.lassobenchfunc.evaluate(x_slice))
            result = np.array(result)
            result = result.reshape(x_shape[:-1])
            return result
