import torch
from torch import Tensor
from typing import Callable, Tuple
import numpy as np
from scipy.optimize import minimize, Bounds
from HDBOBenchmark.algorithms.add_bo_quasi_nt.my_optim.parameter_constraints import (
    make_scipy_bounds,
    _arrayify,
)
from HDBOBenchmark.algorithms.add_bo_quasi_nt.my_optim.utils import columnwise_clamp


def gen_candidates_scipy(
    initial_conditions: Tensor,
    acquisition_function: Callable,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Generate a set of candidates using 'scipy.optimize.minimize'.

    Optimizes an acquisition function starting from a set of initial candidates
    using 'scipy.optimize.minimize' via a numpy converter.

    Args:
        initial_conditions: Starting points for optimization.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.
    """
    ndims = initial_conditions.shape[-1]
    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )  # shape(num_restarts, q, ndims)

    shapeX = clamped_candidates.shape
    x0 = clamped_candidates.view(-1)
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )

    def f_np_wrapper(x: np.ndarray, f: Callable):
        """Given a torch callable, compute value + grad given a numpy array."""
        if np.isnan(x).any():
            raise RuntimeError(
                f"{np.isnan(x).sum()} elements of the {x.size} element array "
                f"`x` are NaN."
            )
        X = (
            torch.from_numpy(x)
            .to(initial_conditions)
            .view(-1, ndims)  # shape(num_restarts x q, ndims)
            .contiguous()
            .requires_grad_(True)
        )
        loss = f(X).sum()
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
        if np.isnan(gradf).any():
            msg = (
                f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                "gradient array `gradf` are NaN. This often indicates numerical issues."
            )
            if initial_conditions.dtype != torch.double:
                msg += " Consider using `dtype=torch.double`."
            raise RuntimeError(msg)
        fval = loss.item()
        return fval, gradf

    x0 = _arrayify(x0)

    def f(x):
        return -acquisition_function(x)

    # To maximize the acquisition_function.
    res = minimize(
        fun=f_np_wrapper, args=(f,), x0=x0, method="L-BFGS-B", jac=True, bounds=bounds
    )
    candidates = torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX)

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates.view(-1, ndims))

    return clamped_candidates, batch_acquisition


def gen_mll_candidates_scipy(
    initial_conditions: Tensor,
    mll: Callable,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
) -> Tuple[Tensor, Tensor]:
    ndims = initial_conditions.shape[-1]
    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )  # shape(num_restarts, q, ndims)
    clamped_candidates = clamped_candidates.reshape(
        (-1, ndims)
    )  # shape(num_restarts * q, ndims)
    candidates = torch.zeros_like(clamped_candidates).to(clamped_candidates)
    scipy_bounds = Bounds(
        lb=lower_bounds.detach().numpy(),
        ub=upper_bounds.detach().numpy(),
        keep_feasible=True,
    )

    def mll_np_wrapper(x: np.ndarray, fun: Callable):
        """Given a torch callable, compute value + grad given a numpy array."""
        X = torch.from_numpy(x).requires_grad_(True)  # (ndims)
        loss = fun(X)
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        gradf = torch.autograd.grad(loss, X)[0].detach().numpy()
        if np.isnan(gradf).any():
            msg = (
                f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                "gradient array `gradf` are NaN. This often indicates numerical issues."
            )
            raise RuntimeError(msg)
        fval = loss.item()
        return fval, gradf

    def neg_mll(t):
        return -mll(t)

    batch_mll_values = torch.zeros(clamped_candidates.shape[0]).to(clamped_candidates)
    for i, x0 in enumerate(clamped_candidates):
        res = minimize(
            fun=mll_np_wrapper,
            args=(neg_mll,),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=scipy_bounds,
        )
        candidates[i] = torch.from_numpy(res.x)
        batch_mll_values[i] = torch.tensor(-res.fun)
    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )
    clamped_candidates = clamped_candidates.reshape(
        initial_conditions.shape
    )  # shape(num_restarts, q, ndims)

    return clamped_candidates, batch_mll_values


def _gen_mll_candidates_scipy(
    initial_conditions: Tensor,
    mll: Callable,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
) -> Tuple[Tensor, Tensor]:
    ndims = initial_conditions.shape[-1]
    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )  # shape(num_restarts, q, ndims)

    shapeX = clamped_candidates.shape
    x0 = clamped_candidates.view(-1)
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )

    def mll_np_wrapper(x: np.ndarray, fun: Callable):
        """Given a torch callable, compute value + grad given a numpy array."""
        if np.isnan(x).any():
            raise RuntimeError(
                f"{np.isnan(x).sum()} elements of the {x.size} element array "
                f"`x` are NaN."
            )
        X = (
            torch.from_numpy(x)
            .to(initial_conditions)
            .view(-1, ndims)  # shape(num_restarts x q, ndims)
            .contiguous()
            .requires_grad_(True)
        )
        loss = torch.tensor(0.0).to(X)
        for xx in X:
            loss += fun(xx)
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
        if np.isnan(gradf).any():
            msg = (
                f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                "gradient array `gradf` are NaN. This often indicates numerical issues."
            )
            if initial_conditions.dtype != torch.double:
                msg += " Consider using `dtype=torch.double`."
            raise RuntimeError(msg)
        fval = loss.item()
        return fval, gradf

    def neg_mll(t):
        return -mll(t)

    res = minimize(
        fun=mll_np_wrapper,
        args=(neg_mll,),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
    )
    candidates = torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX)

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )

    batch_mll_values = torch.zeros(shapeX[0]).to(clamped_candidates)
    for i, xx in enumerate(clamped_candidates.view(-1, ndims)):
        batch_mll_values[i] = mll(xx)

    return clamped_candidates, batch_mll_values
