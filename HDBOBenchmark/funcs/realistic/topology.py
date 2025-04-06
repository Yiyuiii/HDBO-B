"""
==================================================
Repo: https://github.com/ISosnovik/top
==================================================
"""
import numpy as np
from pathlib import Path
from typing import Optional
from HDBOBenchmark.funcs.base.FunctionBase import TestFunction


class Topology(TestFunction):
    def __init__(
        self,
        dim=1600,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=False,
        data_dir="assets/topology/",
    ):
        self.load_data(data_dir)
        super().__init__(1600, "Topology")

        self._x_bound = np.array([[0, 1]] * self.dim, dtype=float)  # shape(dim, 2)
        self.y_scale = 2
        self._min_pt = None
        self._min_val = -1

        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)
        self.switch_maximum()

    def load_data(self, data_dir="assets/topology/"):
        data_dir = Path(data_dir)
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
        cur_data = data_dir / "target_bin.npy"

        if not cur_data.exists():
            print(f"Downloading {str(cur_data)}")
            import requests

            url_dir = "https://raw.githubusercontent.com/Yiyuiii/HDBO-B/master/assets/topology/"
            r = requests.get(url_dir + "target_bin.npy")
            r.raise_for_status()
            with cur_data.open("wb") as f:
                f.write(r.content)
        self.target = np.load(str(cur_data)).flatten()

    @staticmethod
    def score_function(predicted, target, metric: Optional[str] = "cos"):
        if not isinstance(predicted, np.ndarray):
            predicted = np.asarray(predicted).flatten()
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)

        if target.ndim < predicted.ndim:
            target = np.expand_dims(target, axis=0)
        assert target.ndim == predicted.ndim

        if metric == "cos":
            num = float(np.dot(predicted, target))  # 向量点乘
            denom = np.linalg.norm(predicted, axis=-1) * np.linalg.norm(target, axis=-1)
            sim = num / denom
            sim = np.where(denom != 0, sim, 0)
            return sim
        elif metric == "jaccard":
            import scipy as sp

            return sp.spatial.distance.jaccard(predicted, target)
        else:
            raise NotImplementedError(f"Metric {metric} is not implemented.")

    def _evaluate(self, x):  # black-box objective function to minimize
        return self.score_function(x, self.target)
