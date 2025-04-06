def test_func_time():
    import time
    import numpy as np
    from importlib import import_module, reload

    print("Start testing HDBOBenchmark.")
    import HDBOBenchmark

    print("Start testing initilization time.")
    for file in dir(HDBOBenchmark.base):
        try:
            class_name = file

            start_time = time.perf_counter_ns()
            reload(HDBOBenchmark.base)
            submodule = import_module(f"HDBOBenchmark.base.{file}")
            func = getattr(submodule, class_name)(dim=1000)
            time_cost = time.perf_counter_ns() - start_time
            print(f"Inited {class_name}, time {time_cost * 1e-6} ms.")

            x = np.random.random(func.dim)
            start_time = time.perf_counter_ns()
            for _ in range(100):
                func.evaluate(x)
            time_cost = time.perf_counter_ns() - start_time
            print(f"Run 100 * {class_name}, time {time_cost * 1e-6} ms.")
        except Exception as e:
            pass

    print("Start testing HDBOBenchmark.")
    import HDBOBenchmark

    print("Start testing initilization time.")
    start_time = time.perf_counter_ns()
    reload(HDBOBenchmark)
    time_cost = time.perf_counter_ns() - start_time
    print(f"Inited HDBOBenchmark, time {time_cost * 1e-6} ms.")
    try:
        for testset_name in HDBOBenchmark.__all__:
            for testset in getattr(HDBOBenchmark, testset_name):
                for func in testset:
                    x = np.random.random(func.dim)
                    start_time = time.perf_counter_ns()
                    for _ in range(100):
                        func.evaluate(x)
                    time_cost = time.perf_counter_ns() - start_time
                    print(f"Run 100 * {func.name}, time {time_cost * 1e-6} ms.")
    except Exception as e:
        pass

    print("Start testing botorch.")
    import torch
    import botorch.test_functions

    print("Start testing initilization time.")
    start_time = time.perf_counter_ns()
    reload(botorch.test_functions)
    time_cost = time.perf_counter_ns() - start_time
    print(f"Inited botorch.test_functions, time {time_cost * 1e-6} ms.")

    for func_name in dir(botorch.test_functions):
        try:
            func = getattr(botorch.test_functions, func_name)(dim=1000)
            x = torch.rand(func.dim)
            start_time = time.perf_counter_ns()
            for _ in range(100):
                func(x)
            time_cost = time.perf_counter_ns() - start_time
            print(f"Run 100 * {func_name}, time {time_cost * 1e-6} ms.")
        except Exception as e:
            pass

    print("Start testing HEBO.")
    import hebo.benchmarks

    print("Start testing initilization time.")
    start_time = time.perf_counter_ns()
    reload(hebo.benchmarks)
    time_cost = time.perf_counter_ns() - start_time
    print(f"Inited hebo.benchmarks, time {time_cost * 1e-6} ms.")

    for group_name in dir(hebo.benchmarks):
        group = getattr(hebo.benchmarks, group_name)
        for func_name in dir(group):
            try:
                func = getattr(group, func_name)(dim=1000)
                x = func.space.sample()
                start_time = time.perf_counter_ns()
                for _ in range(100):
                    func(x)
                time_cost = time.perf_counter_ns() - start_time
                print(f"Run 100 * {func_name}, time {time_cost * 1e-6} ms.")
            except Exception as e:
                pass


if __name__ == "__main__":
    test_func_time()
