import numpy as np
from HDBOBenchmark.algorithms.add_bo_quasi_nt.BOLibkky.preprocessDecomposition import (
    _HyperParam as HyperParam,
)
from HDBOBenchmark.algorithms.add_bo_quasi_nt import add_gp_bo


def addbo_optimize(func, bounds, n_iterations, addbo_kwargs=None):
    if addbo_kwargs is None:
        addbo_kwargs = dict()

    decomp_strategy = (
        addbo_kwargs["decomp_strategy"]
        if "decomp_strategy" in addbo_kwargs.keys()
        else "partialLearn"
    )
    n_group = (
        addbo_kwargs["n_group"]
        if "n_group" in addbo_kwargs.keys()
        else int(np.ceil(func.dim / 5))
    )

    hyperParam = HyperParam(func.dim, n_group, True)
    hyperParam.decomp_strategy = decomp_strategy
    best_y, best_x, x_history, y_history, best_y_history = add_gp_bo(
        func, bounds, n_iterations, hyperParam
    )
    result = {
        "best_x": best_x,
        "best_y": best_y,
        "x_history": x_history,
        "y_history": y_history,
        "best_y_history": best_y_history,
    }
    return best_x, best_y, result
