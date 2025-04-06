from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Branin import Branin as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Branin"
        # BO
        self.dims = 2
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"]["model_name"] = "gp"  # gp gpy for dim 2


args = Args()
