from typing import Optional
import numpy as np
from .base.FunctionBase import TestFunction


def flatten(lists):
    result = list()
    for o in lists:
        if isinstance(o, list):
            result.extend(flatten(o))
        else:
            result.append(o)
    return result


class AdditiveFunction(TestFunction):
    def __init__(
        self,
        sub_funcs: list,
        name="AdditiveFunction",
        weights=None,
        noisefunc_label_list: Optional[list] = None,
        if_permute=False,
        if_auto_scale=True,
    ):
        """
        Args:
            sub_funcs: list: [func1, func2, ...], Nested is allowed
            weights: list: [w1, w2, ...], Nested is allowed
            if_permute: if permute the order of X
        """
        sub_funcs = flatten(sub_funcs)
        assert len(sub_funcs) > 0
        self.sub_funcs = sub_funcs

        self._dim = list()
        self._x_bound = list()
        for f in sub_funcs:
            self._dim.append(f.dim)
            self._x_bound.append(f.bound)

        self._dim = np.sum(self._dim)
        self._x_bound = np.concatenate(self._x_bound)

        super().__init__(self.dim, name)

        if weights is None:
            self.weights = np.ones(len(sub_funcs))
        else:
            self.weights = flatten(weights)
        assert len(self.weights) == len(sub_funcs)

        if noisefunc_label_list is None:
            self.noisefunc_label_list = [False for _ in range(len(sub_funcs))]
        else:
            self.noisefunc_label_list = flatten(noisefunc_label_list)
        assert len(self.noisefunc_label_list) == len(sub_funcs)

        # if_permute
        self.if_permute = if_permute
        if if_permute:
            self.permute_order = np.random.permutation(self._dim)
            self.invert_order = np.zeros(self._dim).astype(int)
            self.invert_order[self.permute_order] = np.arange(self._dim)

        # extremums
        self._min_pt = list()
        self._min_val = 0
        for f in sub_funcs:
            f_min_pt = f.min_pt
            if f_min_pt is None:
                self._min_pt = None
            if self._min_pt is not None:
                self._min_pt.append(f_min_pt)
            f_min_val = f.min_val
            if f_min_val is None:
                self._min_val = None
            if self._min_val is not None:
                self._min_val += f_min_val

        if self._min_pt is not None:
            self._min_pt = np.concatenate(self._min_pt)
            self._min_pt = self._min_pt[self.invert_order]

        # if_auto_scale
        self.init_scaler()
        self.if_auto_scale = if_auto_scale
        if if_auto_scale:
            weights = 0
            for w, l in zip(self.weights, self.noisefunc_label_list):
                if not l:
                    weights += w
            self.y_scaler[1] /= weights

    def _evaluate(self, x) -> dict:
        result = dict()
        fake_val = 0
        val = 0
        p = 0
        for f, w, l in zip(self.sub_funcs, self.weights, self.noisefunc_label_list):
            dim = f.dim
            y = f(x[p : p + dim]) * w
            val += y
            if l:
                fake_val += y
            p += dim
        result["y"], result["true_y"] = val, val - fake_val
        return result

    def evaluate(self, x) -> dict:
        x = self._x_preprocess(x)
        result = dict()
        result["x"] = x
        if self.if_permute:
            x = x[self.permute_order]
        x = self._cal_descaled(x, self.x_scaler)
        _result = self._evaluate(x)
        y, true_y = self._cal_scaled(_result["y"], self.y_scaler), self._cal_scaled(
            _result["true_y"], self.y_scaler
        )
        if not self.if_minium:
            y, true_y = -y, -true_y
        result["y"], result["true_y"] = y, true_y
        return result

    def __call__(self, x) -> float:
        x = self._x_preprocess(x)
        if self.if_permute:
            x = x[:, self.permute_order]
        x = self._cal_descaled(x, self.x_scaler)
        y = 0
        p = 0
        for f, w in zip(self.sub_funcs, self.weights):
            dim = f.dim
            y += f(x[:, p : p + dim]) * w
            p += dim
        y = self._cal_scaled(y, self.y_scaler)
        if not self.if_minium:
            y = -y
        return y

    def to_torch(self, device="cpu"):
        TestFunction.to_torch(self, device)
        for func in self.sub_funcs:
            func.to_torch(device)
