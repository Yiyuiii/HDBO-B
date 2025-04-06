import torch
from torch import Tensor
from typing import List
from torch.linalg import solve, cholesky
from .augKernel import aug_kernel


def combined_kernel(x1, x2, decomposition, bws, scales):
    """This is the complete kernel k0 = sum_i ki
    Args:
        x1: A 'n1 x ndims' tensor
        x2: A 'n2 x ndims' tensor
        decomposition: list
        bws: A 'numGroups' tensor
        scales: A 'numGroups' tensor
    :Returns
        k: A 'n1 x n2' tensor
    """
    num_groups = len(decomposition)
    n1, n2 = x1.shape[0], x2.shape[0]
    k = torch.zeros(n1, n2, dtype=x1.dtype)
    for i in range(num_groups):
        coord = decomposition[i]
        bw = bws[i]
        scale = scales[i]
        k += aug_kernel(x1, x2, coord, bw, scale)
    return k


def combined_kernel_noise(
    x: Tensor,
    decomposition: List[Tensor],
    bws: Tensor,
    scales: Tensor,
    noises: Tensor,
    common_noise: float,
):
    return combined_kernel(x, x, decomposition, bws, scales) + (
        torch.sum(noises) + common_noise
    ) * torch.eye(x.shape[0], dtype=x.dtype)


def combined_kernel_mus(
    x1: Tensor, x2: Tensor, decomposition: List[Tensor], bws: Tensor, scales: Tensor
):
    """This is the complete kernel k0 = sum_i ki
    Args:
        x1: A 'n1 x ndims' tensor
        x2: A 'n2 x ndims' tensor
        decomposition: list
        bws: A 'numGroups' tensor
        scales: A 'numGroups' tensor
    :Returns
        k: A 'n1 x n2' tensor
    """
    num_groups = len(decomposition)
    n1, n2 = x1.shape[0], x2.shape[0]
    k = torch.zeros(n1, n2).to(x1)
    mus = torch.zeros(num_groups, dtype=torch.double)
    for i in range(num_groups):
        coord = decomposition[i]
        bw = bws[i]
        scale = scales[i]
        k11 = aug_kernel(x1, x2, coord, bw, scale)
        k += k11
        ell = cholesky(k11 + 1e-10 * torch.eye(k11.shape[0]).to(k11))
        temp = solve(ell, torch.ones(ell.shape[0], 1).to(ell))
        mus[i] = torch.squeeze(temp.T @ temp)
    return k, mus


def combined_kernel_noise_mus(
    x: Tensor,
    decomposition: List[Tensor],
    bws: Tensor,
    scales: Tensor,
    noises: Tensor,
    common_noise: float,
):
    kxx, mus = combined_kernel_mus(x, x, decomposition, bws, scales)
    return kxx + (torch.sum(noises) + common_noise) * torch.eye(x.shape[0]), mus
