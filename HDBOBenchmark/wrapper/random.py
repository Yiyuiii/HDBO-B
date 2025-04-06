from tqdm import tqdm
import torch
import numpy as np


def random_optimize(func, bounds, n_iterations, random_kwargs=None):
    lowB, upB = bounds[:, 0], bounds[:, 1]

    pbar = tqdm(total=n_iterations)
    lb, ub = torch.tensor(lowB), torch.tensor(upB)
    X = torch.rand(n_iterations, func.dim)
    X = X * (ub - lb) + lb
    Y = list()
    for i in range(n_iterations):
        y = func(X[i])
        Y.append(y)
        pbar.update()
    if func.if_torch:
        Y = torch.stack(Y, dim=0)
    else:
        Y = np.stack(Y, axis=0)

    X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
    best_x, best_y = X[np.argmax(Y)], np.max(Y)
    result = {"best_x": best_x, "best_y": best_y, "x_history": X, "y_history": Y}
    return best_x, best_y, result
