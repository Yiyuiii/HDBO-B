from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs.realistic.lassobench import LassoBenchmark as get_func


class Args(ArgsTemplate):
    def __init__(self, benchname="synt_simple"):
        super().__init__()
        self.name = "LassoBench"
        # BO
        self.dims = 100
        self.get_func = get_func
        self.func = get_func(
            benchname=benchname
        )  # synt_simple, synt_medium, synt_high, synt_hard, Breast_cancer, Diabetes, Leukemia, DNA, RCV1

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gumbel"  # gp for dim 5, gumbel for dim 10 20


args = Args(benchname="synt_simple")
