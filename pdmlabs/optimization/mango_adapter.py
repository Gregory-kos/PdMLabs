"""Mango optimizer adapters for PdMLabs.

Provides ``MangoAdapter`` (Bayesian, default) and ``MangoRandomAdapter``
(pure random search), both wrapping the built-in ``pdmlabs.mango.Tuner``.
The ``@scheduler.parallel`` decorator is applied *inside* the adapter so
experiment ``execute()`` methods define a raw ``(**params) -> float``
objective with no Mango-specific decoration.
"""

from pdmlabs.mango import Tuner, scheduler
from pdmlabs.optimization.base import BaseOptimizerAdapter


class MangoAdapter(BaseOptimizerAdapter):
    """Adapter for the built-in Mango Bayesian optimizer.

    Wraps :class:`pdmlabs.mango.Tuner` with ``optimizer='Bayesian'``
    (Gaussian Process surrogate + UCB acquisition).
    """

    supports_categorical: bool = True
    _mango_strategy: str = "Bayesian"

    def maximize(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        constraint_fn=None,
    ) -> dict:
        """Maximize via Mango, applying ``scheduler.parallel`` internally."""
        wrapped = scheduler.parallel(n_jobs)(objective_fn)
        conf_dict = {
            "initial_random": initial_random,
            "num_iteration": n_iterations,
            "constraint": constraint_fn,
            "optimizer": self._mango_strategy,
        }
        tuner = Tuner(param_space, wrapped, conf_dict=conf_dict)
        results = tuner.maximize()
        return {
            "best_params": results["best_params"],
            "best_objective": results["best_objective"],
            "params_tried": list(results["params_tried"]),
            "objective_values": list(results["objective_values"]),
        }

    def minimize(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        constraint_fn=None,
    ) -> dict:
        """Minimize via Mango's native ``tuner.minimize()``."""
        wrapped = scheduler.parallel(n_jobs)(objective_fn)
        conf_dict = {
            "initial_random": initial_random,
            "num_iteration": n_iterations,
            "constraint": constraint_fn,
            "optimizer": self._mango_strategy,
        }
        tuner = Tuner(param_space, wrapped, conf_dict=conf_dict)
        results = tuner.minimize()
        return {
            "best_params": results["best_params"],
            "best_objective": results["best_objective"],
            "params_tried": list(results["params_tried"]),
            "objective_values": list(results["objective_values"]),
        }


class MangoRandomAdapter(MangoAdapter):
    """Adapter for Mango in pure random-search mode.

    Sets ``optimizer='Random'`` in the Mango conf_dict, bypassing the
    Gaussian Process surrogate and sampling hyperparameters uniformly.
    """

    _mango_strategy: str = "Random"
