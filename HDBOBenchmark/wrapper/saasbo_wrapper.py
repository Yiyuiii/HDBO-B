import numpy as np
import torch
from HDBOBenchmark.algorithms.saasbo import saasbo


def saasbo_optimize(func, bounds, n_iterations, saasbo_kwargs=None):
    if saasbo_kwargs is None:
        saasbo_kwargs = dict()

    lowB, upB = bounds[:, 0], bounds[:, 1]

    def eval_objective(x):
        """x is assumed to be in [0, 1]^d"""
        lb, ub = torch.tensor(lowB).to(x), torch.tensor(upB).to(x)
        x = lb + (ub - lb) * x
        y = func(x)  # Flip the value for minimization
        return torch.tensor(y).to(x)

    X, Y = saasbo(
        eval_func=eval_objective,
        ndims=func.dim,
        n_iterations=n_iterations,
        **saasbo_kwargs
    )
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    best_x, best_y = X[np.argmax(Y)], np.max(Y)
    result = {"best_x": best_x, "best_y": best_y, "x_history": X, "y_history": Y}
    return best_x, best_y, result
