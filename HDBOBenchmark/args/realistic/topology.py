from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs.realistic.topology import Topology as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Topology"
        # BO
        self.dims = 1600
        self.get_func = get_func
        self.func = get_func(self.dims)  # self.n_group)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gumbel"  # gp for dim 5, gumbel for dim 10 20


args = Args()
