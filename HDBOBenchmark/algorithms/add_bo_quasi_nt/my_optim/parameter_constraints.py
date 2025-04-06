import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union
from scipy.optimize import Bounds


def _arrayify(X: Tensor) -> np.ndarray:
    r"""Convert a torch.Tensor (any dtype or device) to a numpy (double) array.

    Args:
        X: The input tensor.

    Returns:
        A numpy array of double dtype with the same shape and data as `X`.
    """
    return X.cpu().detach().contiguous().double().clone().numpy()


def make_scipy_bounds(
    X: Tensor,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
) -> Optional[Bounds]:
    r"""Creates a scipy Bounds object for optimziation

    Args:
        X: `... x d` tensor
        lower_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.
        upper_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.

    Returns:
        A scipy `Bounds` object if either lower_bounds or upper_bounds is not
        None, and None otherwise.

    Example:
        >>> X = torch.rand(5, 2)
        >>> scipy_bounds = make_scipy_bounds(X, 0.1, 0.8)
    """
    if lower_bounds is None and upper_bounds is None:
        return None

    def _expand(bounds: Union[float, Tensor], X: Tensor, lower: bool) -> Tensor:
        if bounds is None:
            ebounds = torch.full_like(X, float("-inf" if lower else "inf"))
        else:
            if not torch.is_tensor(bounds):
                bounds = torch.tensor(bounds)
            ebounds = bounds.expand_as(X)
        return _arrayify(ebounds).flatten()

    lb = _expand(bounds=lower_bounds, X=X, lower=True)
    ub = _expand(bounds=upper_bounds, X=X, lower=False)
    return Bounds(lb=lb, ub=ub, keep_feasible=True)


def np_make_scipy_bounds(X, lower_bounds, upper_bounds):
    """Creates a scipy Bounds object for optimization

    Args:
        X: '... x d' ndarray
        lower_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.
        upper_bounds: Lower bounds on each column (last dimension) of `X`. If
            this is a single float, then all columns have the same bound.
    Returns:
        A scipy 'Bounds' object if either lower_bounds or upper_bounds is not
        None, and None otherwise.
    """
    if lower_bounds is None and upper_bounds is None:
        return None

    if lower_bounds is None:
        lb = np.full_like(X, -np.inf)
    else:
        lb = np.broadcast_to(lower_bounds, X.shape)
    if upper_bounds is None:
        ub = np.full_like(X, np.inf)
    else:
        ub = np.broadcast_to(upper_bounds, X.shape)
    return Bounds(lb.flatten(), ub.flatten(), keep_feasible=True)
