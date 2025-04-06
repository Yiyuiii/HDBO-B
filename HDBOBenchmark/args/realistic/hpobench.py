from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs.realistic.HPOBench import HPOBench as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "HPOBench"
        # BO

        self.get_func = get_func
        self.func = get_func(
            benchname="ml.XGBoostBenchmark",
            seed=1,
        )  # https://github.com/automl/HPOBench/wiki/Available-Containerized-Benchmarks
        self.dims = self.func.dim
        self.method_kwargs["hebo"]["model_name"] = "gumbel"


args = Args()
