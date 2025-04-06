from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.Trid import Trid as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Trid"
        # BO
        self.dims = 5
        self.get_func = get_func
        self.func = get_func(self.dims)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gp"  # gp fe gpy for dim 5, gumbel fe for dim 10


args = Args()
