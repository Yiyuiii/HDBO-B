from scipy.optimize import linear_sum_assignment
from HDBOBenchmark.algorithms.add_bo_quasi_nt.utils.getHungarianCostMatrixOrth import (
    get_hungarian_cost_matrix,
)


def orth_to_permutation(a, cost_idx=2, pwr=2):
    """
    Takes in an Orthogonal matrix and returns the 'closest' permutation matrix
    :param a:
    :param cost_idx:
    :param pwr:
    :return row_ind, col_ind:
    """
    # Prelims
    d, p = a.shape
    if d != p:
        raise Exception("Matrix is not a square matrix.")
    # First get the Cost matrix for the assignment
    c = get_hungarian_cost_matrix(a, cost_idx, pwr)
    # min_X \sum_{ij} C_{ij} X_{ij}  s.t.  X_{ij}=1 iff row i is assigned to column j
    row_ind, col_ind = linear_sum_assignment(c)
    return row_ind, col_ind
