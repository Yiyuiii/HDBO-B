"""
git clone https://github.com/automl/HPOBench.git
cd HPOBench
pip install .
# for xgboost
pip install .[xgboost]
"""

import numpy as np
import importlib
from HDBOBenchmark.funcs.base.FunctionBase import TestFunction


class HPOBench(TestFunction):
    def __init__(
        self,
        benchname="ml.XGBoostBenchmark",
        task_id=167149,
        dim=None,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=True,
        seed=1,
    ):
        packagename, benchname = benchname.split(".")
        Benchmark = importlib.import_module(
            "hpobench.benchmarks.%s" % packagename
        ).__getattribute__(benchname)

        self.b = Benchmark(task_id=task_id)
        self.config = self.b.get_configuration_space(seed=seed)
        self.seed = seed

        super().__init__(dim=len(self.config), name=benchname)

        self._x_bound = np.array([[-1, 1]] * self.dim, dtype=float)  # shape(dim, 2)
        for i, variable in enumerate(self.config.items()):
            variable = variable[1]
            if "Float," in str(variable):
                self._x_bound[i] = (variable.lower, variable.upper)
            elif "Integer," in str(variable):
                self._x_bound[i] = (variable.lower, variable.upper + 1)
            else:
                raise NotImplementedError
        y_scale_dict = {}
        min_val_dict = {}
        self.y_scale = (
            y_scale_dict[benchname] if benchname in y_scale_dict.keys() else 100
        )
        self._min_pt = None
        self._min_val = (
            min_val_dict[benchname] if benchname in min_val_dict.keys() else 0
        )
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def _evaluate(self, x):  # black-box objective function to minimize
        if x.ndim == 1:
            return self._evaluate_hpobench(x)
        else:
            x_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            result = list()
            for x_slice in x:
                result.append(self._evaluate_hpobench(x_slice))
            result = np.array(result)
            result = result.reshape(x_shape[:-1])
            return result

    def _evaluate_hpobench(self, x) -> float:
        assert len(x) >= self.dim
        for i, name in enumerate(self.config.keys()):
            if i >= self.dim:
                break
            v = x[i]
            if "Integer," in str(self.config[name]):
                v = int(v)
            v = np.clip(v, self.config[name].lower, self.config[name].upper)
            self.config[name].default_value = v
            assert self.config[name].is_legal(
                self.config[name].default_value
            ), f"{self.config[name].default_value} is not legal for {str(self.config[name])}."

        result_dict = self.b.objective_function(
            configuration=self.config.get_default_configuration(),
            fidelity=None,
            rng=self.seed,
        )
        return result_dict["function_value"]
