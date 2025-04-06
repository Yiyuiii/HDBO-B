import numpy as np


def get_hungarian_cost_matrix(a, cost_idx, power=2):
    """
    The objective is to take A - an orthogonal matrix and return the 'closest' permutation matrix.
    We can treat this as an assignment problem where the cost of assigning the Ai to Zk
    (where Z is the output Perm matrix) is as follows:
    cost_idx = 1, Cik = 1 - |Aik|,
    cost_idx = 2, Cik = 1 - |Aik| + \sum_{j=/=i} |Ajk|
    cost_idx = 3, Cik = 1 - |Aik| + \sum_{j=/=i} |Ajk| + \sum_{l=/=k}|Ail|
    :param a:
    :param cost_idx:
    :param power:
    :return:
    """
    # Prelims
    d, p = a.shape
    a_abs = np.abs(a) ** power
    c = 1 - a_abs
    if cost_idx >= 2:
        for i in range(d):
            c1 = np.sum(np.vstack((a_abs[1:i], a_abs[i + 1 :])), axis=0)
            c[i] = c[i] + c1
        if cost_idx == 3:
            for j in range(p):
                c2 = np.sum(np.hstack((a_abs[:, 1:j], a_abs[:, j + 1 :])), axis=1)
                c[:, j] = c[:, j] + c2
    return c
