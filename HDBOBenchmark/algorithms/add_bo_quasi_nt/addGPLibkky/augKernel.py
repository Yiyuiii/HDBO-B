import torch
from HDBOBenchmark.algorithms.add_bo_quasi_nt.utils.Dist2GP import dist2gp


def aug_kernel(x1, x2, coord, bw, scale):
    """Each of small kernels is only affected by a subset of the coordinates.
    So we need to make sure that the output of the kernel only depends on these quantities. This is what this function
    is doing. Note that we are not adding the noise here.
    Args:
        x1: A 'n1 x ndims' tensor
        x2: A 'n2 x ndims' tensor
        coord:
        bw:
        scale:
    Returns:
        k(x1, x2): (n1, n2)
    """
    if x1.shape[1] == len(coord):
        x1sub = x1
    else:
        x1sub = x1[:, coord]
    if x2.shape[1] == len(coord):
        x2sub = x2
    else:
        x2sub = x2[:, coord]

    d = dist2gp(x1sub, x2sub)  # (n1, n2)
    return scale * torch.exp(-0.5 * d / bw**2)
