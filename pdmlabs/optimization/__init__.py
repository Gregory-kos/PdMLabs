"""PdMLabs optimizer abstraction layer.

Optimizer Registry
------------------
``OPTIMIZER_REGISTRY`` is a plain ``dict`` mapping string identifiers to
adapter classes.  Adding a new backend requires only inserting a new entry.

Supported identifiers
~~~~~~~~~~~~~~~~~~~~~

``"mango"``
    Built-in Mango Bayesian optimizer (Gaussian Process surrogate). **Default.**

``"mango_random"``
    Mango in pure random-search mode (``optimizer='Random'`` in conf_dict).

``"smac"``
    SMAC3 ``HyperparameterOptimizationFacade``.
    Requires ``pip install pdmlabs[smac]``.

``"gpyopt"``
    GPyOpt ``BayesianOptimization``.
    Requires ``pip install pdmlabs[gpyopt]``.

``"hyperopt"``
    Hyperopt ``fmin`` with TPE (Tree-structured Parzen Estimator).
    Requires ``pip install pdmlabs[hyperopt]``.
    Sequential only — emits ``UserWarning`` when ``n_jobs > 1``.

``"optuna"``
    Optuna 5 ``TPESampler`` with ``multivariate=True`` and
    ``constant_liar=True`` (both now the Optuna 5.0 defaults).
    Multi-process parallelism via ``joblib.Parallel`` +
    ``JournalStorage(JournalFileBackend)``.
    Requires ``pip install pdmlabs[optuna]`` (``optuna>=5.0.0``).
"""

from pdmlabs.optimization.base import BaseOptimizerAdapter
from pdmlabs.optimization.mango_adapter import MangoAdapter, MangoRandomAdapter
from pdmlabs.optimization.smac_adapter import SMAC3Adapter
from pdmlabs.optimization.gpyopt_adapter import GPyOptAdapter
from pdmlabs.optimization.hyperopt_adapter import HyperoptAdapter
from pdmlabs.optimization.optuna_adapter import OptunaAdapter

OPTIMIZER_REGISTRY: dict[str, type[BaseOptimizerAdapter]] = {
    "mango":        MangoAdapter,
    "mango_random": MangoRandomAdapter,
    "smac":         SMAC3Adapter,
    "gpyopt":       GPyOptAdapter,
    "hyperopt":     HyperoptAdapter,
    "optuna":       OptunaAdapter,
}


def get_optimizer(name: str) -> BaseOptimizerAdapter:
    """Instantiate and return the optimizer adapter for *name*.

    Parameters
    ----------
    name:
        One of the keys in ``OPTIMIZER_REGISTRY``.

    Returns
    -------
    BaseOptimizerAdapter
        A fresh adapter instance ready to call ``maximize()`` or ``minimize()``.

    Raises
    ------
    ValueError
        If *name* is not a registered optimizer identifier.
    """
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Supported identifiers: {sorted(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[name]()


__all__ = ["OPTIMIZER_REGISTRY", "get_optimizer", "BaseOptimizerAdapter"]
