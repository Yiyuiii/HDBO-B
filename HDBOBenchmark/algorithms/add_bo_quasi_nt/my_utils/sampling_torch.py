import torch
from torch import Tensor
from torch.quasirandom import SobolEngine
from typing import Optional


def draw_sobol_samples(
    bounds: Tensor,
    n: int,
    q: int,
    batch_shape: Optional[torch.Size] = None,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Draw qMC samples from the box defined by bounds.

    Args:
        bounds: A `2 x d` dimensional tensor specifying box constraints on a
            `d`-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples. As a best practice, use powers of 2.
        q: The size of each q-batch.
        batch_shape: The batch shape of the samples. If given, returns samples
            of shape `n x batch_shape x q x d`, where each batch is an
            `n x q x d`-dim tensor of qMC samples.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A `n x batch_shape x q x d`-dim tensor of qMC samples from the box
        defined by bounds.

    Example:
        >>> bounds = torch.stack([torch.zeros(3), torch.ones(3)])
        >>> samples = draw_sobol_samples(bounds, 16, 2)
    """
    batch_shape = batch_shape or torch.Size()
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    d = bounds.shape[-1]
    lower = bounds[0]
    rng = bounds[1] - bounds[0]
    sobol_engine = SobolEngine(q * d, scramble=True, seed=seed)
    samples_raw = sobol_engine.draw(batch_size * n, dtype=lower.dtype)
    samples_raw = samples_raw.view(*batch_shape, n, q, d).to(device=lower.device)
    if batch_shape != torch.Size():
        samples_raw = samples_raw.permute(-3, *range(len(batch_shape)), -2, -1)
    return lower + rng * samples_raw


def draw_sobol_samples_(
    bounds: Tensor,
    n: int,
    q: int,
    batch_shape: Optional[torch.Size] = None,
    seed: Optional[int] = None,
) -> Tensor:
    r"""Draw qMC samples from the box defined by bounds.

    Args:
        bounds: A `2 x d` dimensional tensor specifying box constraints on a
            `d`-dimensional space, where bounds[0, :] and bounds[1, :] correspond
            to lower and upper bounds, respectively.
        n: The number of (q-batch) samples. As a best practice, use powers of 2.
        q: The size of each q-batch.
        batch_shape: The batch shape of the samples. If given, returns samples
            of shape `n x batch_shape x q x d`, where each batch is an
            `n x q x d`-dim tensor of qMC samples.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A `n x batch_shape x q x d`-dim tensor of qMC samples from the box
        defined by bounds.

    Example:
        >>> bounds = torch.stack([torch.zeros(3), torch.ones(3)])
        >>> samples = draw_sobol_samples(bounds, 16, 2)
    """
    batch_shape = batch_shape or torch.Size()
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    d = bounds.shape[-1]
    lower = bounds[0]
    rng = bounds[1] - bounds[0]
    sobol_engine = SobolEngine(1, scramble=True, seed=seed)
    samples_raw = sobol_engine.draw(batch_size * n, dtype=lower.dtype)
    samples_raw = torch.repeat_interleave(samples_raw, q * d, dim=1)
    samples_raw = samples_raw.view(*batch_shape, n, q, d).to(device=lower.device)
    if batch_shape != torch.Size():
        samples_raw = samples_raw.permute(-3, *range(len(batch_shape)), -2, -1)
    return lower + rng * samples_raw
