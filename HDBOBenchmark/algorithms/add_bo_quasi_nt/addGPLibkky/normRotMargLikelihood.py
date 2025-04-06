import torch
from torch import Tensor
from typing import Callable, List
from torch.linalg import solve
from torch.linalg import cholesky
from HDBOBenchmark.algorithms.add_bo_quasi_nt.addGPLibkky.combinedKernelNoise import (
    combined_kernel_noise,
)


def combined_rot_mean_func(
    x: Tensor,
    z: Tensor,
    common_mean_func: Callable,
    mean_funcs: List[Callable],
    decomposition: List[Tensor],
):
    """
    The common Mean Func takes in X as its arguments while meanFuncs take in Z = X @ A as its arguments.
    Args:
        x:
        z:
        common_mean_func:
        mean_funcs:
        decomposition:
    Returns:
    """
    mu0 = common_mean_func(x)
    num_groups = len(decomposition)
    for k in range(num_groups):
        coord = decomposition[k]
        mu0 += mean_funcs[k](z[:, coord])
    return mu0


def norm_rot_marg_likelihood(
    sigma_sms: Tensor,
    sigma_prs: Tensor,
    decomposition: List[Tensor],
    a: Tensor,
    train_x: Tensor,
    train_y: Tensor,
    mean_funcs: List[Callable],
    common_mean_func: Callable,
    noises: Tensor,
    common_noise: float,
):
    """
    Returns the normalized marginal likelihood
    Args:
        sigma_sms: A 'numGroups'-dim ndarray, which gives the bandwidth for each kernel.
        sigma_prs: A 'numGroups'-dim ndarray, which gives the scale for each kernel.
        decomposition:
        a: A 'ndims x ndims' ndarray, which is norm rotation transform.
        x: A 'num x ndims' ndarray, which is training set.
        y: A 'num' ndarray, which is target set.
        mean_funcs:
        common_mean_func:
        noises:
        common_noise:
    Returns:
        marg_likelihood:
    """
    num_pts = train_x.shape[0]
    # Apply transformation and compute normalized marginal likelihood
    z = train_x @ a
    k_y = combined_kernel_noise(
        z, decomposition, sigma_sms, sigma_prs, noises, common_noise
    )
    ell = cholesky(k_y)  # Since there is noise, the matrix is nonsingular
    train_y = train_y.unsqueeze(-1)  # (num, 1)
    mean_z = combined_rot_mean_func(
        train_x, z, common_mean_func, mean_funcs, decomposition
    ).reshape(
        -1, 1
    )  # (num, 1)
    y_ = train_y - mean_z
    alpha = solve(ell.T, solve(ell, y_))  # inv(ell ell.T) @ (y-m_x)
    return (
        -0.5 * torch.squeeze(y_.T @ alpha)
        - torch.sum(torch.log(torch.diag(ell)))
        - num_pts / 2 * torch.log(torch.tensor(2 * torch.pi))
    )
