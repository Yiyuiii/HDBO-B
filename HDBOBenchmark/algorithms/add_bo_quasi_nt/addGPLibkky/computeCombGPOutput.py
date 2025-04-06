import torch
from torch import Tensor
from typing import Callable, List
from torch.linalg import solve
from HDBOBenchmark.algorithms.add_bo_quasi_nt.addGPLibkky.combinedKernelNoise import (
    combined_kernel,
)
from HDBOBenchmark.algorithms.add_bo_quasi_nt.addGPLibkky.combinedMeanFunc import (
    combined_mean_func,
)


def compute_comb_gp_output(
    x_test: Tensor,
    x_train: Tensor,
    decomposition: List[Tensor],
    ell: Tensor,
    alpha: Tensor,
    bws: Tensor,
    scales: Tensor,
    mean_funcs: List[Callable],
    common_mean_func: Callable,
):
    """
    Returns the predictive mean and variance for the combined GP.
    Args:
        x_test: testing set.
        x_train: training set.
        decomposition: A list with elements that give the coordinates for each group.
        ell: A 'num x num' tensor, which is the kernel matrix of training set.
        alpha:
        bws: A 'numGroups' tensor, which gives bandwidths for each group.
        scales: A 'numGroups' tensor, which gives scales for each group
        mean_funcs: A list containing meanFunction for each group.
        common_mean_func:
    Returns:
        y_mu, y_std, y_k:
    """
    k12 = combined_kernel(x_train, x_test, decomposition, bws, scales)  # shape(num, 1)
    k22 = combined_kernel(x_test, x_test, decomposition, bws, scales)  # shape(1, 1)

    # Compute the outputs
    y_mu = (
        combined_mean_func(x_test, common_mean_func, mean_funcs, decomposition).reshape(
            -1, 1
        )
        + k12.T @ alpha
    )

    v = solve(ell, k12)
    y_k = k22 - v.T @ v
    y_std = torch.sqrt(y_k.diag())
    return y_mu.squeeze(), y_std.squeeze(), y_k
