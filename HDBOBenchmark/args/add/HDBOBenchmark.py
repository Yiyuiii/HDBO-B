from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs import TestFuncs as func_list


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "HDBOBenchmark"
        # BO
        self.func = func_list


args = Args()
