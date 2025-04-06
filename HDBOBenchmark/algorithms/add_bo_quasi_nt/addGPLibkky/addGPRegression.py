import torch
from torch.linalg import cholesky, solve
from .combinedKernelNoise import combined_kernel_noise
from .combinedMeanFunc import combined_mean_func


def add_gp_regression(train_x, train_y, decomposition, hyper_params):
    """
    A python function for perform GP Regression when the model is additive.
    This performs inference in each individual GP.
    Args:
        x: Training input (num, dim)
        y: Traning output
        decomposition: list containing num_groups elements. Each element is a vector containing the coordinates in
            that group.
        hyper_params: Contains the smoothness (sigma_sms), scale (sigma_prs) and noise (noise0, noises) parameters
            for each GP in the additive model.
    Returns:
        mus, k_posts:
        combined_func_h: a function handle for the combined GP.
        func_hs: array with a function handle for each group.
    """
    # Prelims
    num_groups = len(decomposition)

    # Set the hyperparameters for each GP
    # Common Mean Function
    common_mean_func = hyper_params.common_mean_func

    def meanFunc(arg):
        return torch.zeros(arg.shape[0]).to(arg)

    # Mean Functions for each GP
    mean_funcs = [hyper_params.mean_funcs] * num_groups
    # Common Noise parameter
    common_noise = hyper_params.common_noise  # shape()
    # Bandwith parameters
    sigma_sms = hyper_params.sigma_sms  # shape(numGroups)
    # Scale parameters
    sigma_prs = hyper_params.sigma_prs  # shape(numGroups)
    # Noise Parameters
    noises = hyper_params.noises  # shape(numGroups)

    # Construct the Training Kernel Matrix and Invert it
    k0 = combined_kernel_noise(
        train_x, decomposition, sigma_sms, sigma_prs, noises, common_noise
    )
    # To invert this we need to do the cholesky decomposition
    ell = cholesky(k0)
    # compute alpha
    mean_y = combined_mean_func(
        train_x, common_mean_func, mean_funcs, decomposition
    ).reshape(-1, 1)
    train_y = train_y.unsqueeze(-1)  # (num, 1)

    y_ = train_y - mean_y
    alpha = solve(ell.T, (solve(ell, y_)))

    hyper_params.k0 = k0
    hyper_params.ell = ell
    hyper_params.alpha = alpha
