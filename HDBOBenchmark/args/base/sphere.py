from .template import Args as ArgsTemplate
from HDBOBenchmark import Sphere as get_func


class Args(ArgsTemplate):
    def __init__(self):
        super().__init__()
        self.name = "Sphere"
        self.dims = 5
        self.get_func = get_func
        self.func = get_func(self.dims)

        self.method_kwargs["hebo"][
            "model_name"
        ] = "svgp"  # gumbel, fe, gpy for dim 5 10


args = Args()
