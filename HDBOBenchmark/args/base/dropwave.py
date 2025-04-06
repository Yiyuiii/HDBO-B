from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Dropwave import DropWave as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "DropWave"
        # BO
        self.dims = 10
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gpy_mlp"  # gpy_mlp gumbel for dim 5 10 20, mcbn psgld for dim 20


args = Args()
