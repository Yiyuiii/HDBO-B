from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs import TestFuncs500 as func_list


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "TestFuncs500"
        # BO
        self.func = func_list


args = Args()
