from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Eggholder import EggHolder as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "EggHolder"
        # BO
        self.dims = 2
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"]["model_name"] = "gpy"  # gp


args = Args()
