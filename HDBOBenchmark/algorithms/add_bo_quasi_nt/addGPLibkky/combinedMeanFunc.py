def combined_mean_func(x, common_mean_func, mean_funcs, decomposition):
    """
    The total mean function (obtained by adding the common and individual mean function
    :param x:
    :param common_mean_func:
    :param mean_funcs:
    :param decomposition: list with elements
    :return mu0: (num,)
    """
    mu0 = common_mean_func(x)
    num_groups = len(decomposition)
    for k in range(num_groups):
        coord = decomposition[k]
        mu0 += mean_funcs[k](x[:, coord])
    return mu0
