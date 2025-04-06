import logging

import numpy as np
from numpy.linalg import cholesky


def stable_cholesky(k):
    """
    Sometimes nominally positive semi-definite matrices are not  positive semi-definite matrices due to numerical
    issues. By adding a small value to the diagonal we can make it positive semi-definite. This is what this
    function does.
    Use this iff you know that k should be positive semi-definite. We do not check for errors.
    :param k: matrix
    :return l: lower-triangular Cholesky factor of k
    """
    diag_power = min(np.ceil(np.log10(np.abs(np.min(np.diag(k))))) - 1, -11)
    if not (np.abs(diag_power) < np.inf):
        diag_power = -10

    # Now keep trying until Cholesky is successful
    success = False
    k += 10**diag_power * np.eye(k.shape[0])
    while not success:
        try:
            l = cholesky(k)
            success = True
        except Exception:
            logging.warning("CHOL failed with diag_power = %d\n" % diag_power)
            diag_power += 1
            k += 10**diag_power * np.eye(k.shape[0])
    return l
