from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Hartmann6d import Hartmann6d as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Hartmann6d"
        # BO
        self.dims = 6
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"]["model_name"] = "gpy"  # gpy gp for dim 6


args = Args()
