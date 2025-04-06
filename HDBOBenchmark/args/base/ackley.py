from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Ackley import Ackley as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Ackley"
        # BO
        self.dims = 5
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gp"  # gp for dim 5, gumbel for dim 10 20


args = Args()
