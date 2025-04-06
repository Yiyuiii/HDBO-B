import torch
import math
from tqdm import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, qExpectedImprovement
from botorch.optim import optimize_acqf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class RemBo:
    def __init__(self, d_orig, d_embedding, n_trials, initial_random_samples=10):
        self.initial_random_samples = initial_random_samples
        self.d_embedding = d_embedding
        self.A = torch.randn(size=(d_embedding, d_orig), dtype=dtype, device=device)
        # buffer
        self.train_x = torch.empty(size=(n_trials, d_orig), dtype=dtype, device=device)
        self.x_embedded = torch.empty(
            size=(n_trials, d_embedding), dtype=dtype, device=device
        )
        self.train_y = torch.empty(size=(n_trials, 1), dtype=dtype, device=device)
        self.count = 0

    def select_query_point(self, batch_size=1):
        """
        :param batch_size: the number of query points per time
        :return x_query: (batch_size, d_orig) numpy array
        :return candidate: (batch_size, d_e) numpy array
        """
        embedding_bounds = torch.tensor(
            [[-math.sqrt(self.d_embedding), math.sqrt(self.d_embedding)]]
            * self.d_embedding,
            dtype=dtype,
            device=device,
        )
        if self.count < self.initial_random_samples:
            # Select query point randomly from embedding_boundaries
            candidate = torch.rand(
                size=(1, self.d_embedding), dtype=dtype, device=device
            )
            # manifold
            candidate = (
                candidate * (embedding_bounds[:, 1] - embedding_bounds[:, 0])
                + embedding_bounds[:, 0]
            )
        else:
            train_Y = self.train_y[: self.count]
            train_Y = (train_Y - train_Y.mean()) / (train_Y.std() + 1e-6)
            # fit model
            gp = SingleTaskGP(train_X=self.x_embedded[: self.count], train_Y=train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            # Construct an acquisition function
            ei = qExpectedImprovement(gp, train_Y.max())
            # ucb = UpperConfidenceBound(gp, beta=0.2)
            # Optimize the acquisition function
            embedding_bounds = embedding_bounds.T
            candidate, acq_value = optimize_acqf(
                acq_function=ei,
                bounds=embedding_bounds,
                q=batch_size,
                num_restarts=10,
                raw_samples=512,
            )  # shape(batch_size, d_e)

        # Map to higher dimensional space and clip to hard boundaries [-1, 1]
        x_query = torch.clip(self._manifold_to_dataspace(candidate), min=-1.0, max=1.0)
        return x_query, candidate

    def _manifold_to_dataspace(self, x_embedded):
        """
        Map data from manifold to original data space.

        :param x_embedded: (1 x d_embedding) numpy.array
        :return: (1 x d_orig) numpy.array
        """
        return x_embedded @ self.A

    def save_point(self, x_query, y_query, x_embedded):
        """Update internal model for observed (X, y) from true function.
        The function is meant to be used as follows.
            1. Call 'select_query_point' to update self.X_embedded with a new
                embedded query point, and to return a query point X_query in the
                original (unscaled) search space
            2. Evaluate X_query to get y_query
            3. Call this function ('update') to update the surrogate model (e.g.
                Gaussian Process)

        Args:
            x_query ((1,d_orig) tensor):
                Point in original input space to query
            y_query (float):
                Value of black-box function evaluated at X_query
            x_embedded ((1,d_e) numpy array):
        """
        # add new rows of data
        self.train_x[self.count] = x_query
        self.train_y[self.count] = y_query
        self.x_embedded[self.count] = x_embedded
        self.count += 1


def rembo(eval_objective, dim, em_dim, n_init, total_trials):
    opt = RemBo(dim, em_dim, total_trials, initial_random_samples=n_init)

    pbar = tqdm(total=total_trials)
    for _ in range(total_trials):
        x_queries, x_embedded = opt.select_query_point()

        # Evaluate the batch of query points 1-by-1
        for x_query, x_e in zip(x_queries, x_embedded):
            y_query = eval_objective((x_query + 1) / 2)  # manifold to [0,1]^D
            opt.save_point(x_query, y_query, x_e)
        pbar.update()

    X, Y = (opt.train_x + 1) / 2, opt.train_y
    return X, Y


# if __name__ == "__main__":
#     import numpy as np

#     DIM = 100
#     EM_DIM = 3
#     N_INIT = 5
#     TOTAL_TRIALS = 30
#     from botorch.test_functions import Branin
#     branin = Branin().to(dtype=dtype, device=device)

#     def branin_emb(x):
#         """x is assumed to be in [0, 1]^d"""
#         lb, ub = branin.bounds
#         return branin(lb + (ub - lb) * x[..., :2]) * -1  # Flip the value for minimization

#     X, Y = rembo(branin_emb, D=DIM, d=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS)
#     Y_np = -1 * Y
#     from matplotlib import pyplot as plt

#     fig = plt.figure(figsize=(12, 6))
#     ax = fig.add_subplot(111)
#     ax.grid(alpha=0.2)
#     ax.plot(range(1, 31), np.minimum.accumulate(Y_np))
#     ax.plot([0, len(Y_np)], [0.398, 0.398], "--", c="g", lw=3, label="Optimal value")
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Best objective found')
#     plt.savefig("results.png")
if __name__ == "__main__":
    from botorch.test_functions import Ackley

    true_dim = 10
    fun = Ackley(dim=true_dim).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    DIM = 100
    EM_DIM = 4
    N_INIT = 5
    TOTAL_TRIALS = 30

    def eval_objective(x):
        """This is a helper function we use to unnormalize and evalaute a point"""
        x = x[:true_dim]
        lb, ub = fun.bounds
        return fun(lb + (ub - lb) * x) * -1

    X, Y = rembo(
        eval_objective, dim=DIM, em_dim=EM_DIM, n_init=N_INIT, total_trials=TOTAL_TRIALS
    )
    Y_np = -1 * Y
    import numpy as np
    import matplotlib.pyplot as plt

    fx = np.minimum.accumulate(Y_np)
    plt.plot(fx, marker="", lw=3)

    plt.plot([0, len(Y_np)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
    plt.show()
