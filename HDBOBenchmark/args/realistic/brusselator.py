from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs.realistic.pde import Brusselator as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Brusselator"
        # BO
        self.dims = 4
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gpy"  # gp for dim 5, gumbel for dim 10 20


args = Args()
