from tqdm import tqdm
import torch
from torch.quasirandom import SobolEngine

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf

WARMUP_STEPS = 256
NUM_SAMPLES = 128
THINNING = 16

tkwargs = {
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}


def saasbo(eval_func, ndims, n_iterations, n_init=10, batch_size=1):
    X = SobolEngine(dimension=ndims, scramble=True, seed=0).draw(n_init).to(**tkwargs)
    Y = torch.tensor([eval_func(x) for x in X]).unsqueeze(-1).to(**tkwargs)

    pbar = tqdm(total=n_iterations)
    for i in range(n_iterations):
        train_Y = (Y - Y.mean()) / Y.std()
        gp = SaasFullyBayesianSingleTaskGP(
            train_X=X,
            train_Y=train_Y,
            train_Yvar=torch.full_like(train_Y, 1e-6),
            outcome_transform=Standardize(m=1),
        )
        fit_fully_bayesian_model_nuts(
            gp,
            warmup_steps=WARMUP_STEPS,
            num_samples=NUM_SAMPLES,
            thinning=THINNING,
            disable_progbar=True,
        )

        EI = qExpectedImprovement(model=gp, best_f=train_Y.max())
        candidates, acq_values = optimize_acqf(
            EI,
            bounds=torch.cat((torch.zeros(1, ndims), torch.ones(1, ndims))).to(
                **tkwargs
            ),
            q=batch_size,
            num_restarts=10,
            raw_samples=1024,
        )

        Y_next = torch.cat([eval_func(x).unsqueeze(-1) for x in candidates]).unsqueeze(
            -1
        )
        X = torch.cat((X, candidates))
        Y = torch.cat((Y, Y_next))
        pbar.update()

    return X, Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from botorch.test_functions import Branin

    branin = Branin().to(**tkwargs)

    def branin_emb(x):
        """x is assumed to be in [0, 1]^d"""
        lb, ub = branin.bounds
        return (
            branin(lb + (ub - lb) * x[..., :2]) * -1
        )  # Flip the value for minimization

    DIM = 30
    X, Y = saasbo(
        eval_func=branin_emb, ndims=DIM, n_iterations=8, n_init=10, batch_size=5
    )
    Y_np = -1 * Y.cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.minimum.accumulate(Y_np), color="b", label="SAASBO")
    ax.plot([0, len(Y_np)], [0.398, 0.398], "--", c="g", lw=3, label="Optimal value")
    ax.grid(True)
    ax.set_title(f"Branin, D = {DIM}", fontsize=20)
    ax.set_xlabel("Number of evaluations", fontsize=20)
    ax.set_xlim([0, len(Y_np)])
    ax.set_ylabel("Best value found", fontsize=20)
    ax.set_ylim([0, 8])
    ax.legend(fontsize=18)
    plt.savefig("results.png")
