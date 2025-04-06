class Args:
    def __init__(self):
        self.name = None
        # BO
        self.dims = 10
        self.func = None

        # main
        self.numIterations = 100
        self.n_init_sample = 20

        # HEBO
        # surrogate model: svidkl, svgp, gp, gpy, gpy_mlp, rf, deep_ensemble, psgld, mcbn, masked_deep_ensemble,fe_deep_ensemble,gumbel,catboost
        # surrogate model: gp, gpy, rf, fe_deep_ensemble, gumbel

        self.method_kwargs = dict()
        self.method_kwargs["botorch"] = {
            "n_init": self.n_init_sample,
        }

        self.method_kwargs["hebo"] = {
            "model_name": "gpy",
            "rand_sample": self.n_init_sample,
            # "acq_cls": MACE,
            "es": "nsga2",
            "scramble_seed": None,
        }
        if self.method_kwargs["hebo"]["model_name"] == "gp":
            cfg = {
                "lr": 0.01,
                "num_epochs": 100,
                "verbose": False,
                "noise_lb": 8e-4,
                "pred_likeli": False,
            }
        elif self.method_kwargs["hebo"]["model_name"] == "gpy":
            cfg = {"verbose": False, "warp": True}
        elif self.method_kwargs["hebo"]["model_name"] == "gpy_mlp":
            cfg = {"verbose": False}
        elif self.method_kwargs["hebo"]["model_name"] == "rf":
            cfg = {"n_estimators": 20}
        else:
            cfg = dict()
        self.method_kwargs["hebo"]["model_config"] = cfg

        # TurBO
        self.method_kwargs["turbo"] = {
            "n_init": self.n_init_sample,
            "batch_size": 1,
        }

        # RemBO
        self.method_kwargs["rembo"] = {
            "n_init": self.n_init_sample,
            "em_dim": 4,  # this is strict
        }
        # RemBO
        self.method_kwargs["saasbo"] = {
            "n_init": self.n_init_sample,
            "batch_size": 1,
        }
        # AleBO
        self.method_kwargs["alebo"] = {
            "n_init": self.n_init_sample,
            "d": 7,
        }
