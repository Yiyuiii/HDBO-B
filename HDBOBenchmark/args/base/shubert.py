from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Shubert import Shubert as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Shubert"
        # BO
        self.dims = 10
        self.get_func = get_func
        self.func = get_func(self.dims)

        self.method_kwargs["hebo"]["model_name"] = "gpy"  # gpy for dim 5 10


args = Args()
