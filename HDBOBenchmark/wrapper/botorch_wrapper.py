from math import log2
from tqdm import tqdm
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
import numpy as np
from scipy.stats.qmc import Sobol
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf


def suggest(input_x, input_y, bounds, n_suggestion=1, device="cpu"):
    # Standardize to 0-mean, 1-variance.
    train_y = standardize(input_y)  # shape(num, 1)
    min_x, max_x = torch.min(input_x, dim=0).values, torch.max(input_x, dim=0).values

    # min-max scale
    train_x = (input_x - min_x) / (max_x - min_x)
    scale_bounds = bounds.T
    scale_bounds = (scale_bounds - min_x) / (max_x - min_x)

    # 2. fit a model
    gp = SingleTaskGP(train_x, train_y)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # 3. Construct an acquisition function:
    UCB = UpperConfidenceBound(gp, beta=0.1)

    # 4. Optimize the acquisition function:
    candidate, acq_value = optimize_acqf(
        UCB,
        bounds=scale_bounds,
        q=n_suggestion,
        num_restarts=64,
        raw_samples=64,
    )
    # invert min-max scale
    candidate = candidate * (max_x - min_x) + min_x

    return candidate


def botorch_optimize(func, bounds, n_iterations, method_kwargs=dict()):
    """
    Maximize
    """
    ndims = bounds.shape[0]
    device = method_kwargs["device"] if "device" in method_kwargs.keys() else "cpu"
    num_rnd = (
        method_kwargs["n_init"] if "n_init" in method_kwargs.keys() else 10
    )  # num of random samples

    if not isinstance(bounds, torch.Tensor):
        bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
    # randomly sample in the bounds.
    sampler = Sobol(d=ndims, scramble=True)
    if num_rnd & (num_rnd - 1) == 0:  # There exists integer m such that n=2^m.
        m = int(log2(num_rnd))
        input_x = sampler.random_base2(m)
    else:
        input_x = sampler.random(num_rnd)
    input_x = torch.tensor(input_x, dtype=torch.float32, device=device)
    input_x = (
        input_x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    )  # shape(num_rnd, ndims)

    # Evaluate the value
    output_y = func(input_x)  # shape(num,)
    output_y = torch.tensor(output_y, dtype=torch.float32, device=device)

    # Record the maximal
    history = torch.zeros(n_iterations)
    max_idx = 0
    maxVal = output_y[0]
    history[0] = maxVal

    pbar = tqdm(total=n_iterations)
    pbar.update()
    for i in range(1, num_rnd):
        if output_y[i] > maxVal:
            maxVal = output_y[i]
            max_idx = i
        history[i] = maxVal
        pbar.update()

    for i in np.arange(num_rnd, n_iterations):
        candidate = suggest(
            input_x=input_x,
            input_y=output_y.unsqueeze(-1),
            bounds=bounds,
            n_suggestion=1,
            device=device,
        )
        # evaluate the value of candidate
        value = func(candidate)
        value = torch.tensor(value, dtype=torch.float32, device=device)
        # add them to input and output set.
        input_x = torch.vstack((input_x, candidate))  # shape(num+1, ndims)
        output_y = torch.hstack((output_y, value))  # shape(num+1,)
        # record the maximal
        if value > maxVal:
            maxVal = value
            max_idx = i
        history[i] = maxVal
        pbar.update(1)

    while maxVal.ndim > 0:
        maxVal = maxVal[0]
    maxPt = input_x[max_idx]
    result = {
        "best_x": maxPt,
        "best_y": maxVal,
        "x_history": input_x,
        "y_history": output_y,
        "best_y_history": history,
    }
    return maxPt, maxVal, result
