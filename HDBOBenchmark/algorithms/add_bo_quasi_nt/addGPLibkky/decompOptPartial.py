from HDBOBenchmark.algorithms.add_bo_quasi_nt.utils.getRandPermMat import (
    get_rand_perm_mat,
)


def decomp_opt_partial(func, ndims):
    num_trials = 5
    a = get_rand_perm_mat(ndims).T
    curr_best_val = func(a)
    for i in range(num_trials):
        p = get_rand_perm_mat(ndims).T
        val = func(p)
        if val < curr_best_val:
            a = p
            curr_best_val = val
    return a
