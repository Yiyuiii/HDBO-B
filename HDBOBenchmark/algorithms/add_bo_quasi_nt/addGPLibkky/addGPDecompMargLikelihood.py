import torch
from torch import Tensor
from torch.linalg import qr
from .normRotMargLikelihood import norm_rot_marg_likelihood
from HDBOBenchmark.algorithms.add_bo_quasi_nt.utils.orthToPermutation import (
    orth_to_permutation,
)
from .decompOptPartial import decomp_opt_partial
from HDBOBenchmark.algorithms.add_bo_quasi_nt.my_optim.optimize import optimize_mll


def add_gp_decomp_marg_likelihood(x: Tensor, y: Tensor, decomp, hyper_params):
    """
    This function attempts to find the best Kernel hyper parameters to fit an additive function.
    The kernel parameters include the smoothness and scale paramters and the decomposition.
    Args:
        x: Training data
        y: Training target
        decomp: If the decomposition need not be learned, then this should contain the true decomposition.
            Otherwise, it should contain two fields d (# of dimensions per group) and M (number of groups).
        hyper_params: should have a field called decompStrategy: It should be one of 'known', 'learn', 'random' and 'partialLearn'.
            'known': The decomposition is known and given in decomp. We will optimize according to this.
            'learn': The decomposition is unknown and should be learned.
            'random': Randomly pick a partition at each iteration.
            'partialLearn': Partially learn the decomposition at each iteration by trying out a few and picking the best.
    Returns:
    """
    # Prelims
    ndims = x.shape[1]

    if hyper_params.decomp_strategy != "known":
        if hasattr(decomp, "m"):
            # First create a placeholder for the decomposition
            raise NotImplementedError
        else:  # then decomp is a vector of values with the number of dims in each group.
            m, p = len(decomp), torch.sum(decomp)
            cum_dims = torch.cumsum(decomp, dim=0)
            cum_dims = torch.hstack((torch.IntTensor([0]), cum_dims))
            decomposition = []
            for i in range(m):
                decomposition.append(torch.arange(cum_dims[i], cum_dims[i + 1]))
    else:
        decomposition = decomp
    num_groups = len(decomposition)
    one_vec = torch.ones(num_groups).to(x)

    # Set the Hyperparameters for each GP
    # Common Mean Function
    common_mean_func = hyper_params.common_mean_func
    # Common Noise parameter
    common_noise = hyper_params.common_noise
    # Mean Function for each GP
    mean_funcs = [hyper_params.mean_funcs] * num_groups
    # Noise Parameters
    noises = hyper_params.noises
    # Some parameters for learning the Kernel smoothness and range
    # Define Bounds for optimization of bandwidth and scale
    sigma_sm_bound = hyper_params.sigma_sm_range  # bandwidth bounds
    sigma_pr_bound = hyper_params.sigma_pr_range  # scale bounds
    bounds = torch.log(torch.vstack((sigma_sm_bound, sigma_pr_bound)))

    # Learn the Decomposition
    if hyper_params.decomp_strategy != "known":
        a = torch.randn((ndims, p), dtype=x.dtype)
        a, _ = qr(a)
        # a = torch.eye(ndims, dtype=x.dtype)
        # maximize marginal likelihood
        # Using shgo
        """
        res = shgo(lambda t: -norm_rot_marg_likelihood(np.exp(t[0]) * one_vec, np.exp(t[1]) * one_vec, decomposition, a, x, y, mean_funcs, common_mean_func, noises, common_noise),
                   bounds)
        solution = res.x
        """

        # Using quasi-Newton method
        def mll(t):
            return norm_rot_marg_likelihood(
                torch.exp(t[0]) * one_vec,
                torch.exp(t[1]) * one_vec,
                decomposition,
                a,
                x,
                y,
                mean_funcs,
                common_mean_func,
                noises,
                common_noise,
            )

        opt_params, _ = optimize_mll(
            mll=mll, bounds=bounds.T, q=1, num_restarts=128, raw_samples=128
        )
        opt_params = opt_params.squeeze()

        sigma_sm_opts = torch.exp(opt_params[0]) * one_vec
        sigma_pr_opts = torch.exp(opt_params[1]) * one_vec

        # Now optimize w.r.t a
        if hyper_params.decomp_strategy == "random":
            raise NotImplementedError
        elif hyper_params.decomp_strategy == "partialLearn":
            a = decomp_opt_partial(
                lambda t: -norm_rot_marg_likelihood(
                    sigma_sm_opts,
                    sigma_pr_opts,
                    decomposition,
                    t,
                    x,
                    y,
                    mean_funcs,
                    common_mean_func,
                    noises,
                    common_noise,
                ),
                ndims,
            )
        elif hyper_params.decomp_strategy == "learn":
            raise NotImplementedError
        else:
            raise Exception("Unknown Strategy to handle decomposition\n")
    else:
        # Otherwise, use the given decomposition
        a = torch.eye(ndims).to(x)

    # Learn the Decomposition Ends here

    # Finally, optimize w.r.t sigma and scale again.
    def mll(t):
        return norm_rot_marg_likelihood(
            torch.exp(t[0]) * one_vec,
            torch.exp(t[1]) * one_vec,
            decomposition,
            a,
            x,
            y,
            mean_funcs,
            common_mean_func,
            noises,
            common_noise,
        )

    """
    res = shgo(lambda t: -norm_rot_marg_likelihood(np.exp(t[0]) * one_vec, np.exp(t[1]) * one_vec, decomposition, a, x, y, mean_funcs, common_mean_func, noises, common_noise),
               bounds)
    marg_like_val, opt_params = -res.fun, res.x
    """
    # Using quasi-Newton method
    opt_params, marg_like_val = optimize_mll(
        mll=mll, bounds=bounds.T, q=1, num_restarts=128, raw_samples=128
    )
    opt_params = opt_params.squeeze()

    sigma_sm_opts = torch.exp(opt_params[0]) * one_vec
    sigma_pr_opts = torch.exp(opt_params[1]) * one_vec

    # Finally, return the learned decomposition
    learned_decomp = []
    if hyper_params.decomp_strategy == "partialLearn":
        permute_order = torch.arange(ndims, dtype=torch.long) @ a.long()
        for i in range(num_groups):
            learned_decomp.append(permute_order[decomposition[i]])
    elif hyper_params.decomp_strategy == "random":
        row_ind, col_ind = orth_to_permutation(a.numpy())
        row_ind, col_ind = torch.from_numpy(row_ind), torch.from_numpy(col_ind)
        indices = torch.argsort(col_ind)
        row_ind = row_ind[indices]
        learned_decomp = []
        for i in range(num_groups):
            learned_decomp.append(row_ind[decomposition[i]])
    elif hyper_params.decomp_strategy == "known":
        learned_decomp = decomposition
    else:
        raise NotImplementedError
    """
    row_ind, col_ind = orth_to_permutation(a.numpy())
    row_ind, col_ind = torch.from_numpy(row_ind), torch.from_numpy(col_ind)
    indices = torch.argsort(col_ind)
    row_ind = row_ind[indices]
    learned_decomp = []
    for i in range(num_groups):
        learned_decomp.append(row_ind[decomposition[i]])
    """
    return sigma_sm_opts, sigma_pr_opts, a, learned_decomp, marg_like_val
