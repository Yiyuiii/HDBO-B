from .AdditiveFunction import AdditiveFunction
from .base import *

TestFuncs100 = [
    AdditiveFunction(
        name="0. Sphere(2)_Zero(98)",
        if_permute=True,
        sub_funcs=[Sphere(2), Zero(98)],
        noisefunc_label_list=[False, True],
    ),
    AdditiveFunction(
        name="1. RosenBrock(4)^10_Eggholder(2)^10_Zero(40)",
        if_permute=True,
        sub_funcs=[[[Rosenbrock(4), EggHolder(2)] for _ in range(10)], Zero(40)],
        noisefunc_label_list=[[[False, False] for _ in range(10)], True],
    ),
    AdditiveFunction(
        name="2. ProductSines(50)_RosenBrock(50)",
        if_permute=True,
        sub_funcs=[ProductSines(50), Rosenbrock(50)],
    ),
    AdditiveFunction(
        name="3. Eggholder(2)^50",
        if_permute=True,
        sub_funcs=[EggHolder(2) for _ in range(50)],
    ),
    AdditiveFunction(
        name="4. Rastrigin(100)",
        if_permute=True,
        sub_funcs=[
            Rastrigin(100),
        ],
    ),
    AdditiveFunction(
        name="5. Ackley(10)^10",
        if_permute=True,
        sub_funcs=[Ackley(10) for _ in range(10)],
    ),
    AdditiveFunction(
        name="6. Ackley(20)_ProductSines(20)_Hartmann6d(6)^5_Noise(30)",
        if_permute=True,
        sub_funcs=[
            Ackley(20),
            ProductSines(20),
            [Hartmann6d(6) for _ in range(5)],
            Noise(30),
        ],
        noisefunc_label_list=[False, False, [False for _ in range(5)], True],
    ),
    AdditiveFunction(
        name="7. Ackley(20)_ProductSines(20)_0.5Rastrigin(20)_0.2Hartmann6d(6)^6_Noise(4)",
        if_permute=True,
        sub_funcs=[
            Ackley(20),
            ProductSines(20),
            Rastrigin(20),
            [Hartmann6d(6) for _ in range(6)],
            Noise(4),
        ],
        weights=[1, 1, 0.5, [0.2 for _ in range(6)], 1],
        noisefunc_label_list=[False, False, False, [False for _ in range(6)], True],
    ),
    AdditiveFunction(
        name="8. Ackley(20)_ProductSines(20)_Rastrigin(20)_Hartmann6d(6)^6_Noise(4)",
        if_permute=True,
        sub_funcs=[
            Ackley(20),
            ProductSines(20),
            Rastrigin(20),
            [Hartmann6d(6) for _ in range(6)],
            Noise(4),
        ],
        noisefunc_label_list=[False, False, False, [False for _ in range(6)], True],
    ),
    AdditiveFunction(
        name="9. Hartmann(6)^16_Zero(4)",
        if_permute=True,
        sub_funcs=[[Hartmann6d(6) for _ in range(16)], Zero(4)],
        noisefunc_label_list=[[False for _ in range(16)], True],
    ),
]

TestFuncs500 = [
    AdditiveFunction(
        name="10. Hartmann(6)_Zero(494)",
        if_permute=True,
        sub_funcs=[Hartmann6d(6), Zero(494)],
        noisefunc_label_list=[False, True],
    ),
    AdditiveFunction(
        name="11. RosenBrock(4)^50_Eggholder(2)^50_Zero(200)",
        if_permute=True,
        sub_funcs=[[[Rosenbrock(4), EggHolder(2)] for _ in range(50)], Zero(200)],
        noisefunc_label_list=[[[False, False] for _ in range(50)], True],
    ),
    AdditiveFunction(
        name="12. ProductSines(100)^3_RosenBrock(100)^2",
        if_permute=True,
        sub_funcs=[
            [ProductSines(100) for _ in range(3)],
            [Rosenbrock(100) for _ in range(2)],
        ],
    ),
    AdditiveFunction(
        name="13. Eggholder(2)^250",
        if_permute=True,
        sub_funcs=[EggHolder(2) for _ in range(250)],
    ),
    AdditiveFunction(
        name="14. Rastrigin(500)",
        if_permute=True,
        sub_funcs=[
            Rastrigin(500),
        ],
    ),
    AdditiveFunction(
        name="15. Ackley(10)^50",
        if_permute=True,
        sub_funcs=[Ackley(10) for _ in range(50)],
    ),
    AdditiveFunction(
        name="16. Ackley(20)^5_ProductSines(20)^5_Hartmann6d(6)^25_Noise(150)",
        if_permute=True,
        sub_funcs=[
            [[Ackley(20), ProductSines(20)] for _ in range(5)],
            [Hartmann6d(6) for _ in range(25)],
            Noise(150),
        ],
        noisefunc_label_list=[
            [[False, False] for _ in range(5)],
            [False for _ in range(25)],
            True,
        ],
    ),
    AdditiveFunction(
        name="17. Ackley(20)^5_ProductSines(20)^5_0.5Rastrigin(20)^5_0.2Hartmann6d(6)^30_Noise(20)",
        if_permute=True,
        sub_funcs=[
            [[Ackley(20), ProductSines(20), Rastrigin(20)] for _ in range(5)],
            [Hartmann6d(6) for _ in range(30)],
            Noise(20),
        ],
        weights=[[[1, 1, 0.5] for _ in range(5)], [0.2 for _ in range(30)], 1],
        noisefunc_label_list=[
            [[False, False, False] for _ in range(5)],
            [False for _ in range(30)],
            True,
        ],
    ),
    AdditiveFunction(
        name="18. Ackley(20)^5_ProductSines(20)^5_Rastrigin(20)^5_Hartmann6d(6)^30_Noise(20)",
        if_permute=True,
        sub_funcs=[
            [[Ackley(20), ProductSines(20), Rastrigin(20)] for _ in range(5)],
            [Hartmann6d(6) for _ in range(30)],
            Noise(20),
        ],
        noisefunc_label_list=[
            [[False, False, False] for _ in range(5)],
            [False for _ in range(30)],
            True,
        ],
    ),
    AdditiveFunction(
        name="19. Hartmann(6)^80_Zero(20)",
        if_permute=True,
        sub_funcs=[[Hartmann6d(6) for _ in range(80)], Zero(20)],
        noisefunc_label_list=[[False for _ in range(80)], True],
    ),
]

TestFuncs1000 = [
    AdditiveFunction(
        name="20. Ackley(10)_Zero(990)",
        if_permute=True,
        sub_funcs=[Ackley(10), Zero(990)],
        noisefunc_label_list=[False, True],
    ),
    AdditiveFunction(
        name="21. RosenBrock(4)^100_Eggholder(2)^100_Zero(400)",
        if_permute=True,
        sub_funcs=[[[Rosenbrock(4), EggHolder(2)] for _ in range(100)], Zero(400)],
        noisefunc_label_list=[[[False, False] for _ in range(100)], True],
    ),
    AdditiveFunction(
        name="22. ProductSines(100)^5_RosenBrock(100)^5",
        if_permute=True,
        sub_funcs=[
            [ProductSines(100) for _ in range(5)],
            [Rosenbrock(100) for _ in range(5)],
        ],
    ),
    AdditiveFunction(
        name="23. Eggholder(2)^500",
        if_permute=True,
        sub_funcs=[EggHolder(2) for _ in range(500)],
    ),
    AdditiveFunction(
        name="24. Rastrigin(1000)",
        if_permute=True,
        sub_funcs=[
            Rastrigin(1000),
        ],
    ),
    AdditiveFunction(
        name="25. Ackley(10)^100",
        if_permute=True,
        sub_funcs=[Ackley(10) for _ in range(100)],
    ),
    AdditiveFunction(
        name="26. Ackley(20)^10_ProductSines(20)^10_Hartmann6d(6)^50_Noise(300)",
        if_permute=True,
        sub_funcs=[
            [[Ackley(20), ProductSines(20)] for _ in range(10)],
            [Hartmann6d(6) for _ in range(50)],
            Noise(300),
        ],
        noisefunc_label_list=[
            [[False, False] for _ in range(10)],
            [False for _ in range(50)],
            True,
        ],
    ),
    AdditiveFunction(
        name="27. Ackley(20)^10_ProductSines(20)^10_0.5Rastrigin(20)^10_0.2Hartmann6d(6)^60_Noise(40)",
        if_permute=True,
        sub_funcs=[
            [[Ackley(20), ProductSines(20), Rastrigin(20)] for _ in range(10)],
            [Hartmann6d(6) for _ in range(60)],
            Noise(40),
        ],
        weights=[[[1, 1, 0.5] for _ in range(10)], [0.2 for _ in range(60)], 1],
        noisefunc_label_list=[
            [[False, False, False] for _ in range(10)],
            [False for _ in range(60)],
            True,
        ],
    ),
    AdditiveFunction(
        name="28. Ackley(20)^10_ProductSines(20)^10_Rastrigin(20)^10_Hartmann6d(6)^60_Noise(40)",
        if_permute=True,
        sub_funcs=[
            [[Ackley(20), ProductSines(20), Rastrigin(20)] for _ in range(10)],
            [Hartmann6d(6) for _ in range(60)],
            Noise(40),
        ],
        noisefunc_label_list=[
            [[False, False, False] for _ in range(10)],
            [False for _ in range(60)],
            True,
        ],
    ),
    AdditiveFunction(
        name="29. Hartmann(6)^160_Zero(40)",
        if_permute=True,
        sub_funcs=[[Hartmann6d(6) for _ in range(160)], Zero(40)],
        noisefunc_label_list=[[False for _ in range(160)], True],
    ),
]

TestFuncs = TestFuncs100 + TestFuncs500 + TestFuncs1000

if __name__ == "__main__":
    print(TestFuncs)
