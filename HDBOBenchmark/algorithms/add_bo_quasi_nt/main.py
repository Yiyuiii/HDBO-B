from HDBOBenchmark.utils.logging import logger
from math import log2
from numpy import ndarray
from tqdm import tqdm
import torch
from typing import Callable
from HDBOBenchmark.algorithms.add_bo_quasi_nt.addGPLibkky.addGPRegression import (
    add_gp_regression,
)
from HDBOBenchmark.algorithms.add_bo_quasi_nt.addGPLibkky.addGPDecompMargLikelihood import (
    add_gp_decomp_marg_likelihood,
)
from HDBOBenchmark.algorithms.add_bo_quasi_nt.utils.projectToRectangle import (
    project_to_rectangle,
)
from HDBOBenchmark.algorithms.add_bo_quasi_nt.my_utils.sampling_torch import (
    draw_sobol_samples,
)
from HDBOBenchmark.algorithms.add_bo_quasi_nt.BOLibkky.getNextQueryPoint import (
    get_next_query_pt,
)


def add_gp_bo(
    oracle: Callable,
    bounds: ndarray,
    num_iterations: int,
    params,
):
    """Implements Gaussian Process Bandits/ Bayesian Optimization using Additive Gaussian Processes.
    See ICML 2015 parper: "High Dimensional Bayesian Optimization and Bandits via Additive Models". K.Kandasamy,
    J.Schneider, B.Poczos
    Args:
        oracle: A function handle for the function you wish to optimize.
        bounds: A 'numDims x 2' ndarray specifying the lower and upper bound of each dimension.
        num_iterations: The number of GPB/BO iterations.
        params: A sturcture specifying the various hyper parameters for optimization. If you wish
            to use default settings, pass an empty struct. Otherwise, see demoCustomise.py to see how to set each hyper
            parameter. Also see below.
    Returns:
        maxVal: The maximum queried value of the function.
        maxPt: The queried point with the maximum value.
        boQueries: A matrix indicating the points at which the algorithm queried.
        boVals: A vector of the query values.
        history: A vector of the maximum value obtained up until that iteration.

    params should have a field called decompStrategy: It should be one of 'known', 'learn', 'random' and 'partialLearn'.
    'known': The decomposition is known and given in decomp. We will optimize according to this.
    'learn': The decomposition is unknown and should be learned.
    'random': Randomly pick a partition at each iteration.
    'partialLearn': Partially learn the decomposition at each iteration by trying out a few and picking the best.
    The default is partialLearn and is the best option if you don't know much about your function.
    """

    # Prelims
    if isinstance(bounds, ndarray):
        bounds = torch.from_numpy(bounds)
    ndims = bounds.shape[0]
    max_threshold_exceeds, num_iter_param_relearn = 5, 25
    decomp = params.decomp
    # The Decomposition
    if params.decomp_strategy == "known":
        decomposition = decomp
        num_groups = len(decomposition)
        # do some diagnostics on the decomposition and print them out
        # relevant_coords = torch.hstack(decomposition)
        # num_relevant_coords = len(relevant_coords)
        # if num_relevant_coords != len(torch.unique(relevant_coords)):
        #     raise Exception('The same coordinate cannot appear in different groups')
        # logger.info('# Groups: %d, %d/%d coordinates are relevant\n' % (num_groups, num_relevant_coords, ndims))
    elif hasattr(decomp, "m"):
        # Now decomposition should have two fields d and m
        num_groups = decomp.m
    else:  # in this case the decomposition is given.
        num_groups = len(decomp)

    # Initialization points
    num_init_pts = 2 ** int(log2(num_iterations // 10))
    # num_init_pts = 10
    logger.info("Obtaining %d Points for Initialization.\n" % num_init_pts)
    init_pts = draw_sobol_samples(
        bounds=bounds.T, n=num_init_pts, q=1
    ).squeeze()  # shape(num_init, ndims)

    """init_pts = torch.repeat_interleave(torch.arange(1, num_init_pts+1, dtype=bounds.dtype).reshape(-1, 1) / num_init_pts, ndims, dim=1)
    init_pts = init_pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]"""

    init_vals = torch.tensor([oracle(pt) for pt in init_pts], dtype=init_pts.dtype)

    num_iterations -= num_init_pts

    # The Bandwidth
    # This BO algorithm will set the bandwidth via its own procedure
    al_bw_lb = params.al_bw_lb
    al_bw_ub = params.al_bw_ub
    # Set an initial bandwidth. This will change as the algorithm progresses
    al_curr_bw = al_bw_ub

    # Define the following before proceeding
    bo_queries = torch.vstack(
        (init_pts, torch.zeros(num_iterations, ndims).to(init_pts))
    )
    bo_vals = torch.hstack((init_vals, torch.zeros(num_iterations).to(init_vals)))
    best_x, best_y, history = (
        init_pts[0],
        init_vals[0],
        torch.zeros(num_init_pts + num_iterations).to(bo_vals),
    )
    history[0] = best_y
    for i in torch.arange(1, len(init_vals)):
        if init_vals[i] > best_y:
            best_y = init_vals[i]
            best_x = init_pts[i]
        history[i] = best_y

    thresh_exceeded_counter = 0

    logger.info("Performing BO (dim = %d)\n" % ndims)
    pbar = tqdm(total=num_iterations)
    for bo_iter in range(num_iterations):
        if (bo_iter + 1) % num_iter_param_relearn == 0:
            pbar.set_postfix({"MaxVal": "%0.4f" % best_y})

        # Prelims
        num_x = num_init_pts + bo_iter
        input_x = bo_queries[:num_x]
        output_y = bo_vals[:num_x]
        # standardize to mean 0 and var 1
        y_mean = torch.mean(output_y, dim=0)  # shape(num) => ()
        y_std = torch.std(output_y, dim=0)  # shape()
        train_y = (output_y - y_mean) / y_std  # shape(num)

        train_x = input_x
        scale_bounds = bounds
        """min-max scale
        min_x = torch.min(input_x, dim=0, keepdims=True).values
        max_x = torch.max(input_x, dim=0, keepdims=True).values  # shape(1, ndims)
        train_x = (input_x - min_x) / (max_x - min_x)
        scale_bounds = (bounds - min_x.T) / (max_x.T - min_x.T)
        """

        # First redefine ranges for the GP bandwidth if needed
        if (not params.use_fixed_bandwidth) and (
            bo_iter % num_iter_param_relearn == 0
            or thresh_exceeded_counter == max_threshold_exceeds
        ):
            if thresh_exceeded_counter == max_threshold_exceeds:
                al_bw_ub = max(al_bw_lb, 0.9 * al_curr_bw)
                thresh_exceeded_counter = 0
                logger.warning(
                    "Threshold Exceeded %d times - Reducing BW\n"
                    % max_threshold_exceeds
                )
            else:
                pass
            # Define the BW range for addGPMargLikelihood
            if al_bw_ub == al_bw_lb:
                params.fix_sm = True
                params.sigma_pr_ranges = al_bw_lb * torch.ones(num_groups)
            else:
                params.fix_sm = False
                # Use same bandwidth for now.
                params.use_same_sm = True
                params.sigma_sm_range = torch.DoubleTensor([al_bw_lb, al_bw_ub])

            # Obtain the optimal GP parameters
            if params.decomp_strategy != "stoch1":
                (
                    al_curr_bws,
                    al_curr_scales,
                    _,
                    learned_decomp,
                    marg_like_val,
                ) = add_gp_decomp_marg_likelihood(train_x, train_y, decomp, params)
                al_curr_bw = al_curr_bws[0]  # modify to allow different bandwidths
            else:
                raise NotImplementedError

        # If stochastic pick a current GP
        if params.decomp_strategy != "stoch1":
            params.sigma_sms = al_curr_bws
            params.sigma_prs = al_curr_scales
            curr_iter_decomp = learned_decomp
        else:
            raise NotImplementedError

        # Now build the GP
        add_gp_regression(train_x, train_y, curr_iter_decomp, params)
        # Now obtain the next point
        candidate, _, candidate_std = get_next_query_pt(
            train_x, train_y, params, curr_iter_decomp, scale_bounds
        )
        # If it is too close, perturb it a bit
        while (
            torch.min(
                torch.sqrt(torch.sum((train_x - candidate.unsqueeze(0)) ** 2, dim=1))
            )
            / al_curr_bw
            < 1e-10
        ):
            logger.debug("The candidate is too close to training set, perturb it a bit")
            candidate = project_to_rectangle(
                candidate + 0.1 * al_curr_bw * torch.randn(ndims), scale_bounds
            )
        # Determine the current best point
        """invert min-max scale
        candidate = candidate * (max_x - min_x) + min_x
        """
        candidate = candidate.squeeze()
        value = oracle(candidate)
        if value > best_y:
            best_y = value
            best_x = candidate

        bo_queries[num_init_pts + bo_iter] = candidate
        bo_vals[num_init_pts + bo_iter] = value

        # Check if next_point_std is too small
        if candidate_std < params.opt_pt_std_threshold:
            thresh_exceeded_counter += 1
        else:
            thresh_exceeded_counter = 0
        # Store the current best value
        history[bo_iter + num_init_pts] = best_y
        pbar.update()

    return (
        best_y,
        best_x.numpy(),
        bo_queries.numpy(),
        bo_vals.numpy(),
        history.numpy(),
    )
