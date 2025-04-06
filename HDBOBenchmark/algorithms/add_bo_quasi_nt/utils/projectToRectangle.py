import numpy as np


def project_to_rectangle(pt, bounds=None):
    """
    :param pt: (dims,)
    :param bounds: (dims, 2) ndarray with the lower bounds on the first column and the upper bound on the second column.
        If bounds is not given, it will be taken as [0,1]^d.
    :return:
    """
    dims = pt.shape[0]
    if bounds is None:
        bounds = np.array([[0, 1] * dims])
    below_lbs = (pt - bounds[:, 0]) < 0
    pt[below_lbs] = bounds[below_lbs, 0]

    above_ubs = (pt - bounds[:, 1]) > 0
    pt[above_ubs] = bounds[above_ubs, 1]
    return pt
