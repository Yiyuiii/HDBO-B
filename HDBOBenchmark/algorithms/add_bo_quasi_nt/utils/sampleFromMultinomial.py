import numpy as np


def sample_from_multinomial(p, n=1):
    pre_samples = np.random.multinomial(1, p, size=n)
    samples, _ = np.nonzero(pre_samples)
    return samples
