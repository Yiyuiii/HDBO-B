<!--
 * @Author         : yiyuiii
 * @Date           : 2025-04-06 20:00:00
 * @LastEditors    : yiyuiii
 * @LastEditTime   : 2024-04-06 20:00:00
 * @Description    : None
 * @GitHub         : https://github.com/yiyuiii/HDBO-B
-->

<!-- markdownlint-disable MD033 MD036 MD041 -->

<div align="center">

# HDBO-B: Benchmark for High Dimensional Bayesian Optimization

</div>

<p align="center">
<a href="https://raw.githubusercontent.com/Yiyuiii/HDBO-B/master/LICENSE"><img src="https://img.shields.io/github/license/Yiyuiii/HDBO-B.svg" alt="license"></a>
<a href="https://pypi.python.org/pypi/HDBO-B"><img src="https://img.shields.io/pypi/v/HDBO-B.svg" alt="pypi"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="python">
</p>

**HDBO-B**, a generalized and unified benchmark for High Dimensional Bayesian Optimization, is including testing functions, example algorithms and more examples. 

## :gear: Installation

We recommend using [Miniconda](https://docs.anaconda.com/free/miniconda/) to isolate Python environments and their packages, this protects you from potential package version conflicts.

To install HDBO-B package, choose one below:

- `git clone https://github.com/yiyuiii/HDBO-B && cd HDBO-B && pip install -e .` (You may get example codes **ONLY** by this way.)
- `pip install hdbo-b`

Some realistic tasks and methods may require extra packages, please check `requirements.txt` and instructions in `HDBOBenchmark/funcs/realistic/().py`.

## :rocket: Quick Start

Uncomment lines in `test.py`, and run `python test.py`.

User may also check codes in folder `./example`.

## :wrench: Details

### Test Functions

To import hand-designed 30 test functions: 

```python
from HDBOBenchmark import TestFuncs as func_list
```

We have designed a stricter framework for functions and algorithms, 
This brings a lot of convenience to scaling, while there are a bit more difficulties in getting started.
Check them out in
- `./HDBOBenchmark/base/FunctionBase.py`
- `./HDBOBenchmark/AdditiveFunction.py`
- `./example/wrapper/hebo_wrapper.py`

### Realistic Functions

Available realistic functions including:

(Additional installation may be required to run these library, please check related instructions in our codes.)

- [**MIPLIB**](https://miplib.zib.de/index.html), the real-world pure and mixed integer programs.
> MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library. Mathematical Programming Computation, 2021.

```python
from HDBOBenchmark import MPSModel

func = MPSModel(mps_path='revised-submissions/miplib2010_publically_available/instances/markshare_4_0.mps.gz',
                solu_path='miplib2017-v26.solu')
```

- [**LassoBench**](https://github.com/ksehic/LassoBench), a library for high-dimensional hyperparameter optimization benchmarks based on Weighted Lasso regression.
> Šehić Kenan, Gramfort Alexandre, Salmon Joseph and Nardi Luigi, "LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso", Proceedings of the 1st International Conference on Automated Machine Learning, 2022.

```python
from HDBOBenchmark import LassoBenchmark

func = LassoBenchmark(benchname='synt_simple')
```

- [**py-pde**](https://github.com/zwicker-group/py-pde), a Python package for solving partial differential equations (PDEs).
We implemented the Brusselator with spatial coupling, a realistic calculation problem according to the [official example](https://py-pde.readthedocs.io/en/latest/examples_gallery/pde_brusselator_expression.html) and one existing [setting](https://github.com/bhouri0412/rpn_bo).
For BO, we additionally set the objective as minium the maximum average density of u,v in the grid at the last time t, as in this case the objective is one value.
> https://py-pde.readthedocs.io/en/latest/examples_gallery/pde_brusselator_expression.html
> https://github.com/bhouri0412/rpn_bo

```python
from HDBOBenchmark.funcs.realistic.pde import Brusselator

func = Brusselator()
```

- [**Topology**](https://github.com/ISosnovik/top), The dataset of topology optimization process, used in [T-LBO](https://github.com/huawei-noah/HEBO/tree/master/T-LBO).
> Neural networks for topology optimization, arXiv preprint arXiv:1709.09578, 2017.
> Grosnit, Antoine, et al. "High-Dimensional Bayesian Optimisation with Variational Autoencoders and Deep Metric Learning." arXiv preprint arXiv:2106.03609 (2021).

```python
from HDBOBenchmark import Topology

func = Topology()
```

### Bayesian Optimization Algorithms

Algorithms can be found in `./examples/algorithms` (more original) and `./examples/wrapper` (directly imported by `./examples/optimize.py`), including

- GP(Gaussian Process) + UCB(Upper Confidence Bound) Implementation with [BOTorch](https://github.com/pytorch/botorch)
> M. Balandat, B. Karrer, D. R. Jiang, S. Daulton, B. Letham, A. G. Wilson, and E. Bakshy. BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization. Advances in Neural Information Processing Systems 33, 2020.

- GP/GPy(warped GP)/gumbel/... + MACE(Multi-objective ACquisition function Ensemble) with [HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO)
> Cowen-Rivers, Alexander I., et al. HEBO: Pushing The Limits of Sample-Efficient Hyperparameter Optimisation. Journal of Artificial Intelligence Research, 2022.

- Python implementation of [Add-GP-UCB](https://github.com/kirthevasank/add-gp-bandits)
> Kandasamy K, Schneider J, Póczos B. High dimensional Bayesian optimisation and bandits via additive models. International conference on machine learning, 2015.

- Python implementation of [REMBO](https://github.com/ziyuw/rembo)
> Wang Z, Hutter F, Zoghi M, et al. Bayesian optimization in a billion dimensions via random embeddings. Journal of Artificial Intelligence Research, 2016.

- Wrapper of [ALEBO](https://github.com/facebookresearch/alebo)
> Letham B, Calandra R, Rai A, et al. Re-examining linear embeddings for high-dimensional Bayesian optimization. Advances in neural information processing systems, 2020.

- Wrapper of [TuRBO](https://github.com/uber-research/TuRBO)
> Eriksson D, Pearce M, Gardner J, et al. Scalable global optimization via local bayesian optimization. Advances in neural information processing systems, 2019.

- Wrapper of [SAASBO](https://github.com/martinjankowiak/saasbo)
> Eriksson D, Jankowiak M. High-dimensional Bayesian optimization with sparse axis-aligned subspaces. Uncertainty in Artificial Intelligence, 2021.

`./examples/optimize.py` saves optimizing histories in folder `./result`, which are automatically read by `./example/plot_result.py`.

## :speech_balloon: Common Issues

## :triangular_flag_on_post: TODO List

- [ ] Introducing more realistic test functions

- [ ] Introducing VAE-based algorithms and related datasets needed

## :microscope: Cite Us

## :clipboard: Changelog

#### 2025.04.06 > v0.2.0 :fire:
- Added HPOBench and BBOB synthetic functions. Paper accepted by IJCNN 2025. 

#### 2023.06.06 > v0.1.1
- Fixed logging.

#### 2023.06.03 > v0.1.0
- Initialization.
