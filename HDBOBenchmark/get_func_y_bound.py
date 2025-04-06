import numpy as np
import torch
from .utils.logging import logger


def get_func_y_bound(args, method, n_repeats=5):
    min_value = np.inf
    max_value = -np.inf
    # optimize for min value
    name = args.func.name
    dim = args.func.dim
    logger.info(f"func:{name}, dim:{dim}, optimizer:{method.__name__}")
    args.func.switch_maximum()  # for optimizers for maximum
    method_kwargs = dict()
    if "method_kwargs" in dir(args):
        for method_name in ["botorch", "hebo", "turbo", "rembo", "saasbo", "alebo"]:
            if (method_name in method.__name__) and (
                method_name in args.method_kwargs.keys()
            ):
                method_kwargs = args.method_kwargs[method_name]
                break

    for i_repeats in range(n_repeats):
        _, best_y, _ = method(
            args.func, args.func.bound, args.numIterations, method_kwargs
        )
        if isinstance(best_y, torch.Tensor):
            best_y = best_y.detach().cpu().numpy()
        min_value = min(min_value, best_y)
        logger.info("min_value at iter %d: %.4f." % (i_repeats, min_value))

    # optimize for max value
    args.func.switch_minium()
    for i_repeats in range(n_repeats):
        _, best_y, _ = method(
            args.func, args.func.bound, args.numIterations, method_kwargs
        )
        if isinstance(best_y, torch.Tensor):
            best_y = best_y.detach().cpu().numpy()
        max_value = max(max_value, -best_y)
        logger.info("max_value at iter %d: %.4f." % (i_repeats, max_value))

    return min_value, max_value
