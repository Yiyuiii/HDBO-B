from math import log2
from scipy.stats.qmc import Sobol


def draw_sobol_samples(bounds, n, q, seed=None):
    """Draw qMC samples from the box defined by bounds.

    Args:
        bounds: A '2 x d' dimensional ndarray specifying box constraints on a
            'd'-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples. As a best practice, use powers of 2.
        q: The size of each q-batch.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A 'n x q x d'-dimensional ndarray of qMC samples from the box
        defined by bounds.
    """
    ndims = bounds.shape[-1]
    lower = bounds[0]
    rng = bounds[1] - bounds[0]
    sampler = Sobol(d=q * ndims, scramble=True)
    if n & (n - 1) == 0:  # There exists integer m such that n=2^m.
        m = int(log2(n))
        samples_raw = sampler.random_base2(m)
    else:
        samples_raw = sampler.random(n)
    samples_raw = samples_raw.reshape((n, q, ndims))
    return lower + rng * samples_raw


def draw_sobol_samples_common(bounds, n):
    """Draw qMC samples from the box defined by bounds.

    Args:
        bounds: A '2 x d' dimensional ndarray specifying box constraints on a
            'd'-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples. As a best practice, use powers of 2.

    Returns:
        A 'n x d'-dimensional ndarray of qMC samples from the box
        defined by bounds.
    """
    ndims = bounds.shape[-1]
    lower = bounds[0]
    rng = bounds[1] - bounds[0]
    sampler = Sobol(d=ndims, scramble=True)
    if n & (n - 1) == 0:  # There exists integer m such that n=2^m.
        m = int(log2(n))
        samples_raw = sampler.random_base2(m)
    else:
        samples_raw = sampler.random(n)
    return lower + rng * samples_raw
