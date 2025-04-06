import numpy as np

# pip install ax-platform
from HDBOBenchmark.algorithms.alebo import alebo


def alebo_optimize(func, bounds, n_iterations, alebo_kwargs=None):
    if alebo_kwargs is None:
        alebo_kwargs = dict()

    lb, ub = bounds[:, 0], bounds[:, 1]
    if not "D" in alebo_kwargs:
        alebo_kwargs["D"] = func.dim

    def eval_objective_wrap(parameterization):
        """x is assumed to be in [0, 1]^d"""
        x = np.array(
            [parameterization.get(f"x{i}") for i in range(len(parameterization))]
        )
        x = lb + (ub - lb) * x
        y = func(x)  # Flip the value for minimization
        return {"objective": (y, 0.0)}

    X, Y = alebo(
        eval_func=eval_objective_wrap,
        total_trials=n_iterations,
        **alebo_kwargs,
    )

    best_x, best_y = X[np.argmax(Y)], np.max(Y)
    result = {"best_x": best_x, "best_y": best_y, "x_history": X, "y_history": Y}
    return best_x, best_y, result
