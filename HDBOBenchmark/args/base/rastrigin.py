from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Rastrigin import Rastrigin as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Rastrigin"
        # BO
        self.dims = 5
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gpy"  # gumbel fe mcbn psgld for dim 10, gumbel fe for dim 20


args = Args()
