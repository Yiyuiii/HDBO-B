from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Holdertable import HolderTable as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "HolderTable"
        # BO
        self.dims = 2
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"]["model_name"] = "gpy"


args = Args()
