import torch
from torch import Tensor
from typing import Callable
import warnings
from HDBOBenchmark.algorithms.add_bo_quasi_nt.my_utils.sampling_torch import (
    draw_sobol_samples,
)
from torch.quasirandom import SobolEngine


def initialize_q_batch(X: Tensor, Y: Tensor, n: int, eta=1.0):
    """Heuristic for selecting initial conditions for candidate generation.

    This heuristic selects points from 'X' (without replacement) with probability
    proportional to 'exp(eta * Z)', where 'Z = (Y - mean(Y)) / std(Y)' and 'eta'
    is a temperature parameter.

    Args:
        X: A 'b x q x d' ndarray of 'b' samples of
            'q'-batches from a 'd'-dim feature space. Typically, these are generated using qMC sampling.
        Y: A ndarray of 'b' outcomes associated with the samples.
        n: The number of initial condition to be generated. Must be less than 'b'.
        eta: Temperature parameter for weighting samples.

    Returns:
        A 'n x q x d' ndarray of 'n'-'q'-batch initial
        conditions, where each batch of 'n x q x d' samples is selected independently.
    """
    n_samples = X.shape[0]
    if n > n_samples:
        raise RuntimeError(
            f"n ({n}) cannot be larger than the number of "
            f"provided samples ({n_samples})"
        )
    elif n == n_samples:
        return X

    Ystd = Y.std(dim=0)  # shape()
    if torch.any(Ystd == 0):
        warnings.warn(
            "All acquisition values for raw samples points are the same for "
            "at least one batch. Choosing initial conditions at random."
        )
        return X[torch.randperm(n=n_samples)][
            :n
        ]  # X[np.random.permutation(n_samples)][:n]

    max_idx = torch.argmax(Y, axis=0)  # shape()
    Z = (Y - Y.mean(dim=0)) / Ystd  # shape(b)
    etaZ = eta * Z
    weights = torch.exp(etaZ)
    while torch.isinf(weights).any():
        etaZ *= 0.5
        weights = torch.exp(etaZ)
    idcs = torch.multinomial(
        weights, n
    )  # idcs = np.random.choice(len(weights), n, replace=False, p=weights)
    if max_idx not in idcs:
        idcs[-1] = max_idx
    return X[idcs]


def gen_batch_initial_conditions(
    acq_function: Callable,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options=None,
):
    """Generate a batch of initial conditions for random-restart optimziation.
    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A '2 x d' ndarray of lower and upper bounds.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization.
    Returns:
        A 'num_restart x q x d' ndarray of initial conditions.
    """
    options = options or {}
    seed = options.get("seed")
    effective_dim = bounds.shape[-1] * q

    if effective_dim > SobolEngine.MAXDIM:  # Max dimensionality of Sobol is 21201.
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension 21201. Using iid samples instead."
        )
        X_rnd_nlzd = torch.rand(raw_samples, q, bounds.shape[-1], dtype=bounds.dtype)
        X_rnd = bounds[0] + (bounds[1] - bounds[0]) * X_rnd_nlzd
    else:
        X_rnd = draw_sobol_samples(
            bounds=bounds, n=raw_samples, q=q, seed=seed
        )  # shape(n, q, ndims)
    ndims = bounds.shape[-1]
    X_rnd_batch = X_rnd.reshape(-1, ndims)  # shape(raw_samples * q, ndims)
    Y_rnd = acq_function(X_rnd_batch)  # shape(raw_samples * q)
    batch_initial_conditions = initialize_q_batch(X=X_rnd, Y=Y_rnd, n=num_restarts)
    return batch_initial_conditions


def gen_mll_initial_conditions(
    mll: Callable, bounds: Tensor, q: int, num_restarts: int, raw_samples: int
):
    effective_dim = bounds.shape[-1] * q

    if effective_dim > SobolEngine.MAXDIM:  # Max dimensionality of Sobol is 21201.
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension 21201. Using iid samples instead."
        )
        X_rnd_nlzd = torch.rand(raw_samples, q, bounds.shape[-1], dtype=bounds.dtype)
        X_rnd = bounds[0] + (bounds[1] - bounds[0]) * X_rnd_nlzd
    else:
        X_rnd = draw_sobol_samples(
            bounds=bounds, n=raw_samples, q=q
        )  # shape(n, q, ndims)
    # initialize_q_batch
    return X_rnd
