from typing import Callable
from torch import Tensor
from torch.linalg import solve
from HDBOBenchmark.algorithms.add_bo_quasi_nt.addGPLibkky.augKernel import aug_kernel


def compute_ind_gp_output(
    x_test: Tensor,
    x_train: Tensor,
    coord: Tensor,
    ell: Tensor,
    alpha: Tensor,
    bw: Tensor,
    scale: Tensor,
    mean_func: Callable,
):
    """
    This function returns the predictive mean and GP for an individual GP
    Args:
        x_test: A 'n2 x ndims' tensor, testing set.
        x_train: A 'n1 x ndims' tensor, training set.
        coord: coordinates of variables
        ell: A 'n1 x n1' tensor which is cholesky decompostion of kernel matrix of training set.
        alpha:
        bw:  A tensor which gives the bandwidth for current group.
        scale: A tensor which gives the scale for current group.
        mean_func:
    Returns:
        y_mu, y_std, y_k:
    """
    # compute K12 and K22
    k12 = aug_kernel(x_train, x_test, coord, bw, scale)  # (n1, n2)
    k22 = aug_kernel(x_test, x_test, coord, bw, scale)  # (n2, n2)
    """
    Note that we are not adding noise to K22 here
    If we have already observed at x_te then we need to account for each individual noise.
    But if we are interested in prediction at a new unobserved point then we need not add noise.
    """
    # compute the outputs
    y_mu = mean_func(x_test).reshape(-1, 1) + k12.T @ alpha  # Predictive Mean

    v = solve(ell, k12)  # (n1, n2)
    y_k = k22 - v.T @ v  # Predictive Variance  shape(n2, n2)
    return y_mu.squeeze(), y_k.diag()
