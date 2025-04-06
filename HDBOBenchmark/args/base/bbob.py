from .template import Args as ArgsTemplate
from HDBOBenchmark.funcs.base.BBOB import bbob_func_dict

func_name = "Rastrigin"


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = func_name
        # BO
        self.dims = 10
        self.func = bbob_func_dict[func_name]

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gp"  # gp for dim 5, gumbel for dim 10 20


args = Args()
