import numpy as np
from scipy.stats.qmc import Sobol
from math import log2


class SobolSampler:
    def __init__(self, dim, scramble=True, seed=None):
        self.seed_generator = np.random.default_rng(seed)
        self.get_sampler = lambda: Sobol(
            d=dim,
            scramble=scramble,
            # optimization="random-cd",  # extremely slow
            seed=self.seed_generator.integers(10000),
        )

    def sample(self, n):
        sampler = self.get_sampler()
        if n & (n - 1) == 0:  # There exists integer m such that n=2^m.
            m = int(log2(n))
            x = sampler.random_base2(m=m)
        else:
            x = sampler.random(n)
        x = x * 2 - 1
        return x
