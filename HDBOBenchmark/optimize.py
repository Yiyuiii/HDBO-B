import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .utils.os import save_pickle
from .utils.logging import logger


def optimize_process(args, method):
    name = args.func.name
    dim = args.func.dim
    logger.info(
        f"func:{name}, dim:{dim}, min:{args.func.min_val}, optimizer:{method.__name__}"
    )

    args.func.switch_maximum()  # for optimizers for maximum
    method_kwargs = dict()
    if "method_kwargs" in dir(args):
        for method_name in ["botorch", "hebo", "turbo", "rembo", "saasbo", "alebo"]:
            if (method_name in method.__name__) and (
                method_name in args.method_kwargs.keys()
            ):
                method_kwargs = args.method_kwargs[method_name]
                break

    best_x, best_y, result_dict = method(
        args.func, args.func.bound, args.numIterations, method_kwargs
    )
    if isinstance(best_x, torch.Tensor):
        best_x = best_x.detach().cpu().numpy()
    if isinstance(best_y, torch.Tensor):
        best_y = best_y.detach().cpu().numpy()
    for k, v in result_dict.items():
        if isinstance(v, torch.Tensor):
            result_dict[k] = v.detach().cpu().numpy()
    best_y, result_dict["best_y"] = -best_y, -best_y
    for k in ["y_history", "best_y_history"]:
        if k in result_dict.keys():
            result_dict[k] = -result_dict[k]

    logger.info(f"Best value found is {best_y:.4f} with point {best_x}.")
    save_cnt = 0
    save_dir = f"./result/{method.__name__}/{name}_{dim}_{save_cnt}.pkl"
    while os.path.exists(save_dir):
        save_cnt += 1
        save_dir = f"./result/{method.__name__}/{name}_{dim}_{save_cnt}.pkl"
    save_pickle(save_dir, result_dict)


def basefunc_tests():
    # get arguments
    from .args.base.ackley import args as a1
    from .args.base.bukin import args as a2
    from .args.base.dropwave import args as a3
    from .args.base.eggholder import args as a4
    from .args.base.griewank import args as a5
    from .args.base.hartmann6d import args as a6
    from .args.base.holdertable import args as a7
    from .args.base.product_sines import args as a8
    from .args.base.rastrigin import args as a9
    from .args.base.rosen_brock import args as a10
    from .args.base.shubert import args as a11
    from .args.base.sphere import args as a12
    from .args.base.trid import args as a13

    arg_list = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
    # get methods
    from .wrapper import (
        botorch_optimize,
        hebo_optimize,
        alebo_optimize,
        saasbo_optimize,
        rembo_optimize,
        turbo_optimize,
        addbo_optimize,
    )

    method_list = [
        botorch_optimize,
        hebo_optimize,
        addbo_optimize,
        alebo_optimize,
        saasbo_optimize,
        rembo_optimize,
        turbo_optimize,
    ]

    dims = [
        5,
    ]
    for dim in dims:
        for method in method_list:
            for a in arg_list:
                a.func = a.get_func(dim)
                optimize_process(a, method)


def addfunc_tests():
    # get arguments
    from .args.add.HDBOBenchmark import args as a1

    arg_list = [
        a1,
    ]

    # get methods
    from .wrapper import (
        botorch_optimize,
        hebo_optimize,
        addbo_optimize,
        alebo_optimize,
        saasbo_optimize,
        rembo_optimize,
        turbo_optimize,
    )

    method_list = [
        botorch_optimize,
        hebo_optimize,
        addbo_optimize,
        alebo_optimize,
        saasbo_optimize,
        rembo_optimize,
        turbo_optimize,
    ]

    for method in method_list:
        for a in arg_list:
            if isinstance(a.func, list):
                func_list = a.func
                for f in func_list:
                    a.func = f
                    optimize_process(a, method)
                a.func = func_list
            else:
                optimize_process(a, method)


def single_test():
    # get arguments
    from .args.add.HDBOBenchmark import args as a1

    _func = a1.func
    a1.func = [a1.func[0], a1.func[10], a1.func[20]]

    arg_list = [
        a1,
    ]

    # get methods
    from .wrapper import (
        random_optimize,
        botorch_optimize,
        hebo_optimize,
        addbo_optimize,
        alebo_optimize,
        saasbo_optimize,
        rembo_optimize,
        turbo_optimize,
    )

    method_list = [
        random_optimize,
        rembo_optimize,
        botorch_optimize,
        hebo_optimize,
        addbo_optimize,
        turbo_optimize,
        alebo_optimize,
        saasbo_optimize,
    ]

    for method in method_list:
        for a in arg_list:
            if isinstance(a.func, list):
                func_list = a.func
                for f in func_list:
                    a.func = f
                    optimize_process(a, method)
                a.func = func_list
            else:
                optimize_process(a, method)

    a1.func = _func


def special_test():
    # get arguments
    from .args.realistic.miplib import args as a1

    arg_list = [
        a1,
    ]
    # get methods
    from .wrapper import (
        random_optimize,
        botorch_optimize,
        hebo_optimize,
        alebo_optimize,
        saasbo_optimize,
        rembo_optimize,
        turbo_optimize,
        addbo_optimize,
    )

    method_list = [
        random_optimize,
        botorch_optimize,
        hebo_optimize,
        addbo_optimize,
        alebo_optimize,
        saasbo_optimize,
        rembo_optimize,
        turbo_optimize,
    ]

    for method in method_list:
        for a in arg_list:
            optimize_process(a, method)


if __name__ == "__main__":
    single_test()
