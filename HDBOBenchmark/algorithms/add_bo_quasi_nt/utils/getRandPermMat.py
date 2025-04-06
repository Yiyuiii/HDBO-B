import torch

"""
def get_rand_perm_mat(dims):
    p = np.zeros((dims, dims))
    shuffle_order = np.random.permutation(dims)
    for i in range(dims):
        p[i, shuffle_order[i]] = 1
    return p
"""


def get_rand_perm_mat(ndims: int):
    p = torch.zeros(ndims, ndims, dtype=torch.double)
    shuffle_order = torch.randperm(ndims)
    for i in range(ndims):
        p[i, shuffle_order[i]] = 1
    return p
