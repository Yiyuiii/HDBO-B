from HDBOBenchmark.args.base.template import Args as ArgsTemplate
from HDBOBenchmark.funcs import TestFuncs1000 as func_list


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "TestFuncs1000"
        # BO
        self.func = func_list


args = Args()
