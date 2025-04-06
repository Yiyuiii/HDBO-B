import numpy as np
from tqdm import tqdm
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


n_suggestions = 1


def hebo_optimize(func, bounds, n_iterations, hebo_kwargs=None):
    if hebo_kwargs is None:
        hebo_kwargs = dict()

    ndims = bounds.shape[0]
    params = []
    for i in range(ndims):
        dd = {"type": "num"}
        dd["name"] = "x%d" % i
        dd["lb"] = bounds[i, 0]
        dd["ub"] = bounds[i, 1]
        params.append(dd)
    space = DesignSpace().parse(params)
    opt = HEBO(space, **hebo_kwargs)
    history = np.zeros(n_iterations)
    pbar = tqdm(total=n_iterations)
    for i in range(n_iterations):
        rec = opt.suggest(n_suggestions)
        values = np.zeros((n_suggestions, 1))
        for j in range(n_suggestions):
            values[j, :] = -func(rec.values[j, :])
        opt.observe(rec, values)
        history[i] = -opt.y.min()
        pbar.update()

    best_x, best_y = opt.best_x.values, -opt.best_y
    result = {
        "best_x": best_x,
        "best_y": best_y,
        "x_history": opt.X.values,
        "y_history": -opt.y.squeeze(),
        "best_y_history": history,
    }
    return best_x, best_y, result
