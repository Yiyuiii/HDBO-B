from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Rosenbrock import Rosenbrock as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Rosenbrock"
        # BO
        self.dims = 5
        self.get_func = get_func
        self.func = get_func(self.dims)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gpy"  # gpy, fe, gumbel for dim 5, # gumbel fe for dim 10 20, svgp for dim 20


args = Args()
