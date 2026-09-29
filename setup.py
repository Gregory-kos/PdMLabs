import os
import re

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "pdmlabs.evaluation.anomaly_evaluator",
        ["pdmlabs/evaluation/anomaly_evaluator.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def get_version():
    init_path = os.path.join(os.path.dirname(__file__), "pdmlabs", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read())
        if match:
            return match.group(1)
        raise RuntimeError("Unable to find version string.")

setup(
    name="pdmlabs",
    version=get_version(),
    author="Anastasios Papadopoulos, Apostolos Giannoulidis, DataLab AUTh",
    description="PdMLabs is an open-source Python automated machine learning benchmarking platform designed to navigate industrial time-series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PdM-Labs/PdMLabs",
    project_urls={
        "Documentation": "https://pdm-labs.github.io/PdMLabs/",
        "Source": "https://github.com/PdM-Labs/PdMLabs",
        "Tracker": "https://github.com/PdM-Labs/PdMLabs/issues",
    },
    python_requires=">=3.11",
    license_files=["LICENSE.txt", "NOTICE.txt"],
    packages=find_packages(include=["pdmlabs", "pdmlabs.*"]),
    ext_modules=cythonize(extensions),
    # Floors are the versions exercised by the working `pdmlabs-docs` environment;
    # caps are the next major (next minor for 0.x, where the API is still unstable).
    # Unbounded floors are what let a fresh install pull pandas 3.x, which removed the
    # `freq="H"` alias the docs relied on.
    install_requires=[
        "arch>=8.0.0,<9",
        "auto_mix_prep>=0.2.0,<0.3",
        "celery>=5.6.3,<6",
        "hurst>=0.0.5,<0.1",
        "joblib>=1.5.3,<2",
        "locket>=1.0.0,<2",
        "matplotlib>=3.10.9,<4",
        "mlflow>=3.13.0,<4",
        "mypy_extensions>=1.1.0,<2",
        "numpy>=2.4.6,<3",
        "pandas>=2.3.3,<3",
        "patsy>=1.0.2,<2",
        # Not a free choice: scikit-survival 0.28 declares scikit-learn<1.10,>=1.9.0.
        "scikit_learn>=1.9.1,<1.10",
        "scipy>=1.17.1,<2",
        "six>=1.17.0,<2",
        "statsmodels>=0.14.6,<0.15",
        "tqdm>=4.68.2,<5",
        "tsfresh>=0.21.2,<0.22",
        "tslearn>=0.8.1,<0.9",
        "scikit-survival>=0.28.0,<0.29",
        # Core, not optional: pdmlabs/method/xgboost.py and xgboostRUL.py import it at
        # module level, so test/test_supervised.py and test/test_rul.py cannot run without it.
        "xgboost>=3.2.0,<4",
    ],
    extras_require={
        # >=2.4.1 is required, not merely preferred. SMAC 2.4.0 and older do
        # `from sklearn.tree._tree import DTYPE`, which scikit-learn 1.9 removed,
        # and scikit-survival (a core dependency) pins scikit-learn to >=1.9,<1.10 --
        # so there is no scikit-learn that satisfies both. SMAC 2.4.1 dropped the
        # reference to that deprecated alias, which resolves the conflict.
        "smac":     ["smac>=2.4.1,<3"],
        "gpyopt":   ["gpyopt", "GPy>=1.0.8,<2"],
        "hyperopt": ["hyperopt>=0.3.0,<0.4"],
        "optuna":   ["optuna>=5.0.0,<6"],
        # Retained as an empty alias so `pip install pdmlabs[xgboost]`, which the
        # published 0.0.3 supports, keeps working now that xgboost is a core dependency.
        "xgboost":  [],
        # nvidia-ml-py, not pynvml: codecarbon 3.x warns that the pynvml
        # distribution is deprecated and that nvidia-ml-py supersedes it.
        "energy":   ["codecarbon>=3.0,<4", "nvidia-ml-py>=12.0,<13", "psutil>=5.9,<8", "pyarrow>=15.0,<25"],
        # Documentation toolchain, used by .github/workflows/docs.yml.
        "docs":     ["sphinx>=8.1.3,<9", "sphinx-book-theme>=1.1.4,<2",
                     "sphinx-design>=0.6.1,<0.7", "sphinx-copybutton>=0.5.2,<0.6"],
        # Everything a contributor needs: all optional backends plus the docs and
        # packaging toolchain. `pip install -e '.[dev]'` is the documented dev setup.
        "dev": [
            "smac>=2.4.1,<3",
            "gpyopt", "GPy>=1.0.8,<2",
            "hyperopt>=0.3.0,<0.4",
            "optuna>=5.0.0,<6",
            "codecarbon>=3.0,<4", "nvidia-ml-py>=12.0,<13", "psutil>=5.9,<8", "pyarrow>=15.0,<25",
            "sphinx>=8.1.3,<9", "sphinx-book-theme>=1.1.4,<2",
            "sphinx-design>=0.6.1,<0.7", "sphinx-copybutton>=0.5.2,<0.6",
            "build>=1.2,<2", "twine>=5.0,<7",
            # Build isolation supplies Cython for a normal `pip install`, but a
            # contributor building with --no-build-isolation needs it in the env.
            "Cython>=3.2.9,<4",
        ],
        # Installs every optional backend in a single resolution step, so pip solves
        # the whole set at once instead of up/downgrading shared dependencies across
        # separate commands. Keep in sync with the individual extras above.
        "all": [
            "smac>=2.4.1,<3",
            "gpyopt", "GPy>=1.0.8,<2",
            "hyperopt>=0.3.0,<0.4",
            "optuna>=5.0.0,<6",
            "codecarbon>=3.0,<4", "nvidia-ml-py>=12.0,<13", "psutil>=5.9,<8", "pyarrow>=15.0,<25",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)

