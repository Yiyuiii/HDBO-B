from setuptools import setup

setup(
    name="HDBOBenchmark",
    version="0.1.2",
    install_requires=[
        "botorch>=0.8.5",
        "numpy>=1.21.5",
        "scipy>=1.9.1",
        "matplotlib>=3.5.2",
        "pillow>=9.2.0",
        "seaborn>=0.13.1",
        "requests>=2.31.0",
    ],
    extras_require={
        "hebo": ["hebo>=0.3.5"],
        "alebo": ["ax-platform>=0.3.2"],
        "pde": ["py-pde"],
        "mip": ["pyscipopt"],
    },
    packages=[
        "HDBOBenchmark",
    ],
    url="https://github.com/Yiyuiii/HDBO-B",
    license="MIT license",
    author="yiyuiii",
    author_email="yiyuiii@foxmail.com",
    description="HDBOBenchmark",
)
