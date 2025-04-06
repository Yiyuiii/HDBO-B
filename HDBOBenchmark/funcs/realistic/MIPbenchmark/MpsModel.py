"""
==================================================
MIPLIB 2017
URL: https://miplib.zib.de/index.html
==================================================
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
from pyscipopt import Model, __version__ as ver_pyscipopt
from HDBOBenchmark.funcs.base.FunctionBase import TestFunction
from .util import download_miplib, get_solutions
from .parse_para_utils import parse_scip_para


class MPSModel(TestFunction):
    def __init__(
        self,
        mps_path,
        solu_path="miplib2017-v26.solu",
        asset_path=r"assets/miplib/",
        dim=None,
        x_scaler=None,
        y_scaler=None,
        auto_init_x_scaler=True,
        auto_init_y_scaler=False,
        solver="scip",
        t_max=7200,  # 2 hours
        seed=None,
        *args,
        **kwargs
    ):
        """
        Args:
            mps_path: the relative path of .mps task.
            solu_path: the relative path of .solu.
            asset_path: the asset path.
            solver: "scip" to optimize the hyperparameters of SCIP solver, others to optimize the original objective of mps.
            t_max: max solve time(seconds) for solvers.
        """
        assert (
            mps_path is not None
        ), "You need to read .mps file to load the problem first!"
        TestFunction.__init__(self)
        self.asset_path = Path(asset_path)
        self.mps_path = mps_path
        self.solu_path = self.asset_path / solu_path
        self.check_assets()
        self.solver = solver
        self.t_max = t_max
        self.seed = seed
        self.getMPSModel(mps_path, t_max, seed)

        self._x_bound = np.array([[-1, 1]] * self.dim, dtype=float)  # shape(dim, 2)
        self._min_pt = None
        self._if_torch = False
        if solver == "scip":
            self.init_SCIP()
        else:
            self.init_raw()
        if dim is not None:
            self._dim = min(dim, self.dim)
        self.init_scaler(x_scaler, y_scaler, auto_init_x_scaler, auto_init_y_scaler)

    def check_assets(self):
        if not (self.asset_path / "revised-submissions").exists():
            download_miplib(path=self.asset_path.resolve(), target="benchmark.zip")
        download_miplib(path=self.asset_path.resolve(), target="miplib2017-v26.solu")

    def getMPSModel(self, mps_path=None, t_max=None, seed=None):
        mps_path = self.mps_path if mps_path is None else mps_path
        t_max = self.t_max if t_max is None else t_max
        seed = self.seed if seed is None else seed
        mpsModel = Model()
        mpsModel.hideOutput(quiet=True)
        mpsModel.readProblem((self.asset_path / mps_path).resolve())
        mpsModel.setParam("display/verblevel", 0)
        mpsModel.setParam("limits/time", t_max)
        if seed:
            mpsModel.setParam("randomization/randomseedshift", seed)
        self.mpsModel = mpsModel

        self._name = self.mpsModel.getProbName()
        if self.solver == "scip":
            self._dim = 135
        else:
            self._dim = self.mpsModel.getNVars()
        # self.if_minium = True if mpsModel.getObjectiveSense() == "minimize" else False
        y_scale_dict = defaultdict(
            lambda: 100, {"markshare_4_0": 1e18, "enlight_hard": 1e31}
        )
        self.y_scale = y_scale_dict[self.name]
        self.solution = get_solutions(self.solu_path.resolve())[self.name]
        self._min_val = float(self.solution["answer"])

    def init_raw(self):
        self.__vars = []
        self.objective_coefficient_list = []
        self.constraint_param_list = []
        # parse variables
        variables = self.mpsModel.getVars()
        type_map = {
            "BINARY": "bool",
            "INTEGER": "int",
            "IMPLINT": "int",
            "CONTINUOUS": "continuous",
        }
        var_list = []
        objective_coefficient_list = []
        for var in variables:
            vtype = var.vtype()
            lb = np.NINF if var.getLbOriginal() == -1e20 else var.getLbOriginal()
            ub = np.Inf if var.getUbOriginal() == 1e20 else var.getUbOriginal()
            index = var.getIndex()
            objective_coefficient = var.getObj()

            var_list.append(
                {
                    "name": str(var),
                    "type": type_map[vtype],
                    "lb": lb,
                    "ub": ub,
                    "index": index,
                }
            )
            self._x_bound[index] = np.array([lb, ub])

            objective_coefficient_list.append(objective_coefficient)

        self.__vars = sorted(var_list, key=lambda d: d["index"])
        self.objective_coefficient_list = objective_coefficient_list
        self.dict_varname_num = dict.fromkeys(
            [var["name"] for var in self.variables], 0.0
        )

        # parse constraints
        conss = self.mpsModel.getConss()
        constraint_param_list = (
            []
        )  # each item is a dict, {'x1': xxx, 'x2': xxx, ... , 'lhs': xxx, 'rhs': xxx}
        unconstraint_conss = []
        for cons in conss:
            item = self.mpsModel.getValsLinear(cons) if cons.isLinear() else {}
            item["lhs"] = self.mpsModel.getLhs(cons)  # lhs < xxx < rhs
            item["rhs"] = self.mpsModel.getRhs(cons)
            constraint_param_list.append(item)
            coefficient_dict = {var.name: var.getLPSol() for var in self.mpsModel.getVars()}
            coefficient_dict.update(
                self.mpsModel.getValsLinear(cons) if cons.isLinear() else {}
            )
            unconstraint_conss.append(
                {"coefficient": list(coefficient_dict.values()), "rhs": item["rhs"]}
            )
            unconstraint_conss.append(
                {
                    "coefficient": (
                        -1 * np.array(list(coefficient_dict.values()))
                    ).tolist(),
                    "rhs": -1 * item["lhs"],
                }
            )
        self.constraint_param_list = constraint_param_list
        self.unconstraint_conss = unconstraint_conss

        # bound
        self._x_bound = np.array([[-1, 1]] * self.dim)  # shape(dim, 2)
        self._x_bound_is_pow = np.array([[0, 0]] * self.dim)  # shape(dim, 2)
        for msg_var in self.variables:
            lb, ub = msg_var["lb"], msg_var["ub"]
            if lb == -np.inf:
                self._x_bound[msg_var["index"], 0] = -100 + np.min(ub, 0)
                self._x_bound_is_pow[msg_var["index"], 0] = 1
            else:
                self._x_bound[msg_var["index"], 0] = lb
            if ub == np.inf:
                self._x_bound[msg_var["index"], 1] = 100 + np.max(lb, 0)
                self._x_bound_is_pow[msg_var["index"], 1] = 1
            else:
                self._x_bound[msg_var["index"], 1] = ub
        self._evaluate = self._evaluate_raw

    def init_SCIP(self):
        design_space, _ = parse_scip_para(Path(__file__).parent / "paras_extended.csv")
        self.params = list(design_space.paras.values())
        self._dim = len(self.params)
        self._x_bound = np.array([[-1, 1]] * self.dim, dtype=float)  # shape(dim, 2)
        self.lb, self.ub = np.empty(self.dim), np.empty(self.dim)
        for i, param in enumerate(self.params):
            self._x_bound[i, 0] = param.lb
            self._x_bound[i, 1] = param.ub
            if param.is_discrete:
                self._x_bound[i, 1] += 1
        self._evaluate = self._evaluate_by_SCIPsolver
        # For time result
        self._min_val = 0
        self.y_scale = float(self.t_max * 2)

    def _evaluate_unconstraint_conss(self, x):
        M = 80
        loss = 0
        for unconstraint_cons in self.unconstraint_conss:
            fn = np.maximum if self.if_minium else np.minimum
            loss += fn(
                np.sum(unconstraint_cons["coefficient"] * x) - unconstraint_cons["rhs"],
                0,
            )
        return M * loss

    def _evaluate_raw(self, x):
        if len(x.shape) > 1:
            x = x[0]
        for i in range(self.dim):
            if self._x_bound_is_pow[i, 0] == 1:
                bias = np.min(self._x_bound[i, 1], 0)
                if x[i] < bias:
                    x[i] = -np.float_power(2, bias - x[i]) + bias
            if self._x_bound_is_pow[i, 1] == 1:
                bias = np.max(self._x_bound[i, 0], 0)
                if x[i] > bias:
                    x[i] = np.float_power(2, x[i] - bias) + bias
            if self.variables[i]["type"] == "bool":
                x[i] = 0 if x[i] < 0.5 else 1

        sum_original_objective = np.sum(
            self.objective_coefficient * x
        )  # Element-wise product
        sum_constraint = self._evaluate_unconstraint_conss(x)
        return sum_original_objective + sum_constraint

    def _evaluate_by_SCIPsolver(self, x_batch):
        if not isinstance(x_batch, np.ndarray):
            x_batch = np.array(x_batch)
        if x_batch.ndim == 1:
            x_batch = x_batch[np.newaxis]
        result = np.zeros(x_batch.shape[0])
        for i_batch, x in enumerate(x_batch):
            # setParam
            self.getMPSModel()
            mpsModel = self.mpsModel
            assert len(x) >= self.dim
            for i, param in enumerate(self.params):
                if i >= self.dim:
                    break
                k = param.name
                v = x[i]
                if not isinstance(v, str):
                    if param.is_discrete:
                        v = int(v)
                    if param.is_categorical:
                        v = param.categories[min(v, param.ub)]
                try:
                    mpsModel.setParam(k, v)
                except KeyError:
                    k = k.split("/")[-1]
                    if ver_pyscipopt <= "3.5.0":
                        k = "separating/" + k
                    else:
                        k = "cutselection/hybrid/" + k
                    mpsModel.setParam(k, v)

            mpsModel.optimize()
            status = mpsModel.getStatus()
            if status == "optimal":
                opt_time = mpsModel.getSolvingTime()
                obj_val = mpsModel.getObjVal()
            else:
                if status == "infeasible":
                    opt_time = self.y_scale
                    obj_val = 1e30
                else:
                    try:
                        obj_val = mpsModel.getObjVal()
                        if abs(obj_val - float(self.solution["answer"])) < 1e-12:
                            opt_time = min(mpsModel.getSolvingTime(), self.t_max)
                        else:
                            opt_time = self.y_scale
                    except:
                        obj_val = 1e30
                        opt_time = self.y_scale
            result[i_batch] = opt_time
        return result
        """
        opt_time = mpsModel.getSolvingTime()
        n_sols = mpsModel.getNSols()
        sol = mpsModel.getBestSol()
        primal_bound = mpsModel.getPrimalbound()
        dual_bound = mpsModel.getDualbound()
        gap = mpsModel.getGap()
        """

    @property
    def variables(self):
        return self.__vars

    @property
    def objective_coefficient(self):
        return self.objective_coefficient_list

    @property
    def constraint_params(self):
        # Each item is a dict: {'x1': xxx, 'x2': xxx, ... , 'lhs': xxx, 'rhs': xxx}
        return self.constraint_param_list
