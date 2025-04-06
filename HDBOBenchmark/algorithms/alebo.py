# Define a function to be optimized.
# Here we use a simple synthetic function with a d=2 true linear embedding, in
# a D=100 ambient space.

import numpy as np

# pip install ax-platform
from ax.modelbridge.strategies.alebo import ALEBOStrategy


def alebo(eval_func, D, d, n_init, total_trials, **kwargs):

    # Define the parameters in the format expected by Ax.
    parameters = [
        {"name": f"x{i}", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"}
        for i in range(D)
    ]
    # Setup the ALEBO optimization strategy
    alebo_strategy = ALEBOStrategy(D=D, d=d, init_size=n_init)

    # Run the optimization loop with that strategy
    # This will take about 30 mins to run

    from ax.service.managed_loop import optimize

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        objective_name="objective",
        evaluation_function=eval_func,
        total_trials=total_trials,
        generation_strategy=alebo_strategy,
    )
    Y = np.array([trial.objective_mean for trial in experiment.trials.values()])
    X = np.array(
        [
            np.array([trial.arm.parameters.get(f"x{i}") for i in range(D)])
            for trial in experiment.trials.values()
        ]
    )
    return X, Y


if __name__ == "__main__":
    from ax.utils.measurement.synthetic_functions import branin
    import numpy as np

    DIM = 100
    EM_DIM = 4

    def branin_evaluation_function(parameterization):
        # Evaluates Branin on the first two parameters of the parameterization.
        # Other parameters are unused.
        lb, ub = np.full((DIM,), -5), np.full((DIM,), 10)
        lb[1], ub[1] = 0, 15
        x = np.array(
            [parameterization.get(f"x{i}") for i in range(len(parameterization))]
        )
        x = lb + (ub - lb) * x
        return {"objective": (-1 * branin(x[..., :2]), 0.0)}

    X, Y = alebo(branin_evaluation_function, D=DIM, d=EM_DIM, n_init=5, total_trials=30)
    Y_np = -1 * Y
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.grid(alpha=0.2)
    ax.plot(range(1, 31), np.minimum.accumulate(Y_np))
    ax.axhline(y=branin.fmin, ls="--", c="k")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best objective found")
    plt.savefig("results.png")
