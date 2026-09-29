"""Hyperopt TPE optimizer adapter for PdMLabs.

All hyperopt imports are lazy so this module is importable even when
hyperopt is not installed. Install with::

    pip install pdmlabs[hyperopt]

The adapter uses ``hyperopt.fmin`` with ``tpe.suggest`` (Tree-structured
Parzen Estimator) as the single entry point.  ``initial_random`` is
forwarded to TPE as ``n_startup_jobs`` via ``functools.partial``, so the
first ``initial_random`` evaluations are pure random before the density
estimator kicks in.  ``max_evals`` is always the hard cap on total
evaluations regardless of ``n_startup_jobs``.

Space conversion rules (all preserve exact candidate sets via ``hp.choice``):

* ``list[int | float | bool | mixed]`` -> ``hp.choice(name, values)``
* ``list[str]``                        -> ``hp.choice(name, values)``
* ``rv_frozen``                        -> ``hp.uniform(name, ppf(0.01), ppf(0.99))``

Parallelism note
----------------
TPE is a strictly sequential density-estimation algorithm.  Each candidate
is proposed based on all previous observed results; there is no batch-
proposal step (unlike GPyOpt's ``local_penalization``).  ``hyperopt.fmin``
does not support multi-process parallelism without external infrastructure
(Spark / MongoDB).

``n_jobs`` is accepted for API consistency but ignored.  A warning
is emitted when ``n_jobs > 1``, directing users to ``optimizer='mango'``,
``'gpyopt'``, or ``'smac'`` for parallel HPO.
"""

import logging
import functools
import numpy as np
from pdmlabs.optimization.base import BaseOptimizerAdapter

_log = logging.getLogger(__name__)


class HyperoptAdapter(BaseOptimizerAdapter):
    """Adapter for Hyperopt TPE via ``hyperopt.fmin``.

    Requires ``hyperopt >= 0.3.0`` (``pip install pdmlabs[hyperopt]``).

    Parallelism
    -----------
    Hyperopt's TPE is strictly sequential; ``n_jobs > 1`` emits a
    warning and falls back to a single worker.  Use
    ``optimizer='mango'``, ``'gpyopt'``, or ``'smac'`` for parallel HPO.
    """

    supports_categorical: bool = True

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def maximize(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        constraint_fn=None,
    ) -> dict:
        """Maximise *objective_fn* using Hyperopt TPE."""
        if n_jobs > 1:
            _log.warning(
                "HyperoptAdapter: n_jobs > 1 is not supported "
                "(TPE is sequential by design). Running with a single "
                "worker. For parallel HPO use optimizer='mango', "
                "'gpyopt', or 'smac'.",
            )
        return self._run(
            param_space, objective_fn, n_iterations, initial_random,
            negate=True,
        )

    def minimize(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        constraint_fn=None,
    ) -> dict:
        """Minimise *objective_fn* using Hyperopt TPE."""
        if n_jobs > 1:
            _log.warning(
                "HyperoptAdapter: n_jobs > 1 is not supported "
                "(TPE is sequential by design). Running with a single "
                "worker. For parallel HPO use optimizer='mango', "
                "'gpyopt', or 'smac'.",
            )
        return self._run(
            param_space, objective_fn, n_iterations, initial_random,
            negate=False,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        initial_random: int,
        negate: bool,
    ) -> dict:
        """Common Hyperopt execution path used by both maximize and minimize.

        Parameters
        ----------
        param_space:
            PdMLabs native hyperparameter search space dict.
        objective_fn:
            Raw ``(**params) -> float`` callable (no decorator required).
        n_iterations:
            Passed to ``fmin`` as ``max_evals`` — the hard cap on **total**
            evaluations (random warm-up + TPE proposals combined).
        initial_random:
            Number of purely random evaluations before the density estimator
            activates.  Forwarded to ``tpe.suggest`` as ``n_startup_jobs``
            via ``functools.partial``.  Must be < ``n_iterations``; if
            ``initial_random >= n_iterations`` TPE never proposes and all
            evaluations are random.
        negate:
            If ``True`` the objective is negated before passing to ``fmin``
            (which always minimises), enabling maximisation.

        Returns
        -------
        dict
            ``{'best_params', 'best_objective', 'params_tried',
            'objective_values'}`` -- values in the original (un-negated)
            direction of the caller.
        """
        from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval  # lazy

        sign = -1.0 if negate else 1.0
        space = self._build_space(param_space)

        # Bind n_startup_jobs so that the first `initial_random` evaluations
        # are pure random before the density estimator kicks in.
        algo = functools.partial(tpe.suggest, n_startup_jobs=max(1, initial_random))

        def hyperopt_target(params: dict) -> dict:
            score = sign * float(objective_fn(**params))
            return {"loss": score, "status": STATUS_OK}

        trials = Trials()
        best_index_dict = fmin(
            fn=hyperopt_target,
            space=space,
            algo=algo,
            max_evals=n_iterations,
            trials=trials,
            rstate=np.random.default_rng(self.random_state),
            show_progressbar=False,
        )
        best_params = space_eval(space, best_index_dict)

        # Extract full evaluation history
        params_tried: list[dict] = []
        objective_values: list[float] = []
        for t in trials.trials:
            if t["result"]["status"] == STATUS_OK:
                trial_vals = {
                    k: v[0] for k, v in t["misc"]["vals"].items() if v
                }
                decoded = space_eval(space, trial_vals)
                params_tried.append(decoded)                    
                raw_loss = t["result"]["loss"]                                  
                # raw_loss == sign * true_score (set inside hyperopt_target).
                # Recovering true_score: sign * (sign * s) = s.
                objective_values.append(sign * raw_loss)

        # Derive best_objective from history for consistency with SMAC/GPyOpt
        if objective_values:
            best_objective = (
                max(objective_values) if negate else min(objective_values)
            )
        else:
            best_objective = 0.0

        return {
            "best_params": best_params,
            "best_objective": best_objective,
            "params_tried": params_tried,
            "objective_values": objective_values,
        }

    # ------------------------------------------------------------------ #
    # Space construction                                                   #
    # ------------------------------------------------------------------ #

    def _build_space(self, param_space: dict) -> dict:
        """Translate PdMLabs param_space to a Hyperopt space dict.

        All list types (``int``, ``float``, ``bool``, ``str``, mixed) map
        to ``hp.choice`` to preserve exact candidate sets.  Continuous
        distributions map to ``hp.uniform``.

        Notes
        -----
        * ``hp.choice`` resolves the index to the actual value inside the
          objective automatically -- no special casting needed (unlike
          GPyOpt's discrete float domain that requires ``int(float(val))``
          or ``bool(round(val))``).
        * Only ``fmin``'s **return dict** contains raw indices;
          ``space_eval(space, best)`` converts them back to original values.
        """
        from hyperopt import hp                                           # lazy
        from scipy.stats._distn_infrastructure import rv_frozen           # lazy

        space: dict = {}
        for name, values in param_space.items():
            if isinstance(values, rv_frozen):
                space[name] = hp.uniform(
                    name,
                    float(values.ppf(0.01)),
                    float(values.ppf(0.99)),
                )
            elif isinstance(values, list):
                # hp.choice preserves exact candidate set for all list types:               
                # int, float, bool, str, or mixed. Hyperopt passes the actual
                # value (not the index) directly into the objective function.
                space[name] = hp.choice(name, values)
            else:
                raise ValueError(
                    f"Unsupported param_space type for '{name}': {type(values)}"
                )
        return space
