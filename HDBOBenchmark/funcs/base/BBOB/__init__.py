from . import bbob
from .wrapper import VizierBBOBWrapper

bbob_func_names = (
    "Rastrigin",
    "LinearSlope",
    "AttractiveSector",
    "StepEllipsoidal",
    "RosenbrockRotated",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    "Weierstrass",
    "SchaffersF7",
    "SchaffersF7IllConditioned",
    "GriewankRosenbrock",
    "Katsuura",
    "Lunacek",
    "Gallagher101Me",
    "Gallagher21Me",
    "NegativeSphere",
    "NegativeMinDifference",
)

bbob_dim = 10
bbob_func_dict = dict()  # minimize
for name in bbob_func_names:
    bbob_func_dict[name] = VizierBBOBWrapper(func=getattr(bbob, name), dim=bbob_dim)
