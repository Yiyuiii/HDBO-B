from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Sin import Sin as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Sin"
        # BO
        self.dims = 2
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gpy"  # fe gp for dim 5, gumbel fe for dim 10, gumbel for dim 20,


args = Args()
