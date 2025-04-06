import torch
from torch import Tensor
from typing import Callable
from .initializers import gen_batch_initial_conditions, gen_mll_initial_conditions
from HDBOBenchmark.algorithms.add_bo_quasi_nt.my_generation.gen import (
    gen_candidates_scipy,
    gen_mll_candidates_scipy,
)


def optimize_acqf(
    acq_function: Callable,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    return_best_only=True,
):
    """
    Args:
        acq_function: An AcquisitionFunction.
        bounds: A '2 x d' ndarray of lower and upper bounds for each column of 'X'.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.

    Returns:
        A two-element tuple containing

        - a 'num_restarts x q x d'-dim ndarray of generated candidates.
        - a 'num_restarts'-dim ndarray of associated acquisition values.
    """
    batch_initial_conditions = gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    batch_candidates, batch_acq_values = gen_candidates_scipy(
        initial_conditions=batch_initial_conditions,
        acquisition_function=acq_function,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
    )
    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]  # shape(q, ndims)
        batch_acq_values = batch_acq_values[best]  # shape()

    return batch_candidates, batch_acq_values


def optimize_mll(
    mll: Callable,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    return_best_only=True,
):
    batch_initial_conditions = gen_mll_initial_conditions(
        mll=mll, bounds=bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples
    )
    batch_candidates, batch_mll_values = gen_mll_candidates_scipy(
        initial_conditions=batch_initial_conditions,
        mll=mll,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
    )
    if return_best_only:
        best = torch.argmax(batch_mll_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]  # shape(q, ndims)
        batch_mll_values = batch_mll_values[best]  # shape()

    return batch_candidates, batch_mll_values
