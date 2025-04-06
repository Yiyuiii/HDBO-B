from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs.realistic.MIPbenchmark.MpsModel import MPSModel as get_func

# enlight_hard sp150x300d neos-787933 nu25-pr12 neos-1122047 neos-1171448 neos-827175

class SCIPArgs(ArgsTemplate):
    def __init__(self, mps_path = "enlight_hard", t_max=20):
        super().__init__()
        self.name = "miplib"
        # BO
        self.dims = 135
        self.get_func = get_func
        # enlight_hard,
        self.func = get_func(
            mps_path=f"revised-submissions/miplib2010_publically_available/instances/{mps_path}.mps.gz",
            solu_path="miplib2017-v26.solu",
            solver="scip",
            t_max=t_max,  # upto 2 hours (7200)
            auto_init_y_scaler=False,
        )

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gumbel"  # gp for dim 5, gumbel for dim 10 20


class RawArgs(ArgsTemplate):
    def __init__(self, mps_path = "enlight_hard"):
        super().__init__()
        self.name = "miplib"
        # BO
        self.dims = 34
        self.get_func = get_func
        # enlight_hard,
        self.func = get_func(
            mps_path=f"revised-submissions/miplib2010_publically_available/instances/{mps_path}.mps.gz",
            solu_path="miplib2017-v26.solu",
            solver=None,
        )

        self.method_kwargs["hebo"][
            "model_name"
        ] = "gumbel"  # gp for dim 5, gumbel for dim 10 20


args = SCIPArgs(mps_path="enlight_hard")
