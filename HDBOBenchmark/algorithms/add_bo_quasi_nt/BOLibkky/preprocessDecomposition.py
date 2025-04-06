import torch


class _HyperParam:
    def __init__(self, ndims: int, num_groups: int, add_remaining_dims=True):
        # First set the Decomposition
        self.decomp_strategy = "partialLearn"
        ndims_per_group = ndims // num_groups
        rem_dims = ndims - ndims_per_group * num_groups

        decomp = ndims_per_group * torch.ones(num_groups).long()

        if rem_dims != 0:
            decomp = torch.hstack(
                (
                    decomp,
                    torch.tensor([rem_dims]),
                )
            )
            num_groups += 1

        self.decomp = decomp
        self.num_groups = num_groups
        # Hyper Parameters for optimization
        # Utility (Acquisition) Function - use UCB as default
        self.utility_func = "UCB"
        # The not-so-important hyper parameters
        self.opt_pt_std_threshold = 0.002

        # GP Hyper parameter for Regression
        # The Common Mean Function
        self.common_mean_func = lambda arg: torch.zeros(arg.shape[0]).to(
            arg
        )  # By default will use all zeros.
        # The Mean function for the individual GPs
        self.mean_funcs = lambda arg: torch.zeros(arg.shape[0]).to(arg)
        # Common noise parameters
        self.common_noise = 1e-2
        # Individual noise
        self.noises = torch.zeros(num_groups, dtype=torch.double)
        # Scale parameters
        self.fix_pr = False
        self.use_same_pr = True
        self.sigma_pr_range = torch.DoubleTensor([0.03, 30.0])
        self.sigma_pr_ranges = []  # use same prior by default
        # Bandwidth parameters
        self.use_fixed_bandwidth = False
        self.fix_sm = False
        self.use_same_sm = True
        self.al_bw_lb = 1e-5
        self.al_bw_ub = 5


class HyperParam:
    def __init__(self, ndims, ndims_per_group, add_remaining_dims=True):
        # First set the Decomposition
        self.decomp_strategy = "partialLearn"
        if add_remaining_dims:
            num_groups = torch.ceil(torch.tensor(ndims / ndims_per_group)).long()
            rem_dims = ndims - ndims_per_group * (num_groups - 1)
            if rem_dims == 0:
                add_remaining_dims = False
        else:
            num_groups = ndims // ndims_per_group

        # Determine the decomposition accordingly.
        if self.decomp_strategy == "known":
            decomp = []
            for i in range(num_groups):
                decomp.append(
                    torch.arange(
                        i * ndims_per_group,
                        min((i + 1) * ndims_per_group, ndims),
                        dtype=torch.long,
                    )
                )
        elif add_remaining_dims:
            decomp = torch.hstack(
                (ndims_per_group * torch.ones(num_groups - 1).long(), rem_dims)
            )
        else:
            raise NotImplementedError
        self.decomp = decomp
        self.num_groups = num_groups
        # Hyper Parameters for optimization
        # Utility (Acquisition) Function - use UCB as default
        self.utility_func = "UCB"
        # The not-so-important hyper parameters
        self.opt_pt_std_threshold = 0.002

        # GP Hyper parameter for Regression
        # The Common Mean Function
        self.common_mean_func = lambda arg: torch.zeros(arg.shape[0]).to(
            arg
        )  # By default will use all zeros.
        # The Mean function for the individual GPs
        self.mean_funcs = lambda arg: torch.zeros(arg.shape[0]).to(arg)
        # Common noise parameters
        self.common_noise = 1e-2
        # Individual noise
        self.noises = torch.zeros(num_groups, dtype=torch.double)
        # Scale parameters
        self.fix_pr = False
        self.use_same_pr = True
        self.sigma_pr_range = torch.DoubleTensor([0.03, 30.0])
        self.sigma_pr_ranges = []  # use same prior by default
        # Bandwidth parameters
        self.use_fixed_bandwidth = False
        self.fix_sm = False
        self.use_same_sm = True
        self.al_bw_lb = 1e-5
        self.al_bw_ub = 5
