import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_plot_func():
    from HDBOBenchmark.plot_func import plot_func
    from HDBOBenchmark.funcs.base import Base_func_list

    for Func in Base_func_list:
        func = Func(dim=2)
        if func.dim == 2:
            plot_func(Func)


def test_plot_result():
    from HDBOBenchmark.plot_result import plot_result

    folder_path = "./result"
    file_list = os.listdir(folder_path)
    method_list = file_list
    # method_list = ['gp_botorch', 'gpy', 'rf', 'gumbel', 'fe_deep_ensemble', 'alebo', 'rembo', 'saasbo', 'turbo']

    # Benchmark functions
    from HDBOBenchmark.args.add.HDBOBenchmark import args

    func_list = args.func
    # func_list = [args.func[0], args.func[10], args.func[20]]

    # from HDBOBenchmark.args.realistic.miplib import RawArgs
    # from HDBOBenchmark.args.realistic.lassobench import Args
    # func_list = [RawArgs(mps_path = "enlight_hard").func, RawArgs(mps_path = "markshare_4_0").func, Args(benchname="synt_simple").func, Args(benchname="synt_high").func]

    # Base functions
    # from example.args.base.ackley import args as a1
    # from example.args.base.bukin import args as a2
    # from example.args.base.dropwave import args as a3
    # from example.args.base.eggholder import args as a4
    # from example.args.base.griewank import args as a5
    # from example.args.base.hartmann6d import args as a6
    # from example.args.base.holdertable import args as a7
    # from example.args.base.product_sines import args as a8
    # from example.args.base.rastrigin import args as a9
    # from example.args.base.rosen_brock import args as a10
    # from example.args.base.shubert import args as a11
    # from example.args.base.sphere import args as a12
    # from example.args.base.trid import args as a13
    # from example.args.base.sin import args as a14
    # arg_list = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14]
    # func_list = list()
    # for a in arg_list:
    #     func_list.append(a.get_func(dim=2))

    plot_result(func_list, method_list, result_path=folder_path)


def test_time():
    from HDBOBenchmark.test_time import test_func_time

    test_func_time()


def test_optimize():
    from HDBOBenchmark.optimize import single_test, special_test

    for _ in range(10):
        special_test()


def move_result():
    from HDBOBenchmark.plot_result import move_result

    dir = r"D:\Codes\HDBOBenchmark"

    move_result(
        dir + r"\result-bak\turbo_optimize",
        dir + r"\result\TuRBO",
    )


def test_func_y_bound():
    from HDBOBenchmark.get_func_y_bound import get_func_y_bound
    from HDBOBenchmark.wrapper import botorch_optimize
    from HDBOBenchmark.args.base.bbob import args

    get_func_y_bound(args, method=botorch_optimize)



if __name__ == "__main__":
    test_plot_func()
    # test_func_y_bound()
    # test_time()
    # move_result()
    # test_optimize()
    # test_plot_result()
