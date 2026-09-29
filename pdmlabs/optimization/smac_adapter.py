"""SMAC3 optimizer adapter for PdMLabs.

All ``smac`` and ``ConfigSpace`` imports are **lazy** (inside method bodies)
so this module is importable even when ``smac`` is not installed.

Install SMAC3 support with::

    pip install pdmlabs[smac]

The adapter uses ``HyperparameterOptimizationFacade`` as the single entry
point. Hyperparameter space conversion rules:

* ``list`` with **string** values  → ``CategoricalHyperparameter``
* ``list`` with **numeric** values → ``OrdinalHyperparameter``
  (preserves the exact candidate set defined by the user)
* ``scipy.stats`` frozen dist      → ``UniformFloatHyperparameter``

SMAC needs a scratch directory to run. It is created under ``$TMPDIR`` (never
in the caller's working directory) and removed when the run ends -- on success,
on exception and on ``KeyboardInterrupt`` alike. ``mkdtemp`` makes it unique
against every process on the machine, so parallel or repeated runs cannot
collide.

Set ``PDMLABS_SMAC_OUTPUT_DIR`` to create that directory under a base of your
choosing **and keep it** after the run, when you need SMAC's ``runhistory.json``
and ``scenario.json`` to debug a bad optimization. Since deletion makes the
location irrelevant, one variable controls both.

Requires ``smac >= 2.4.1``: SMAC 2.4.0 and older import ``DTYPE`` from
``sklearn.tree._tree``, which scikit-learn 1.9 removed, and scikit-survival
pins scikit-learn to ``>=1.9,<1.10``.
"""

import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path

from pdmlabs.optimization.base import BaseOptimizerAdapter

_log = logging.getLogger(__name__)

#: Set to create SMAC's scratch directory under this base and retain it after the run.
OUTPUT_DIR_ENV_VAR = "PDMLABS_SMAC_OUTPUT_DIR"


class SMAC3Adapter(BaseOptimizerAdapter):
    """Adapter for SMAC3 ``HyperparameterOptimizationFacade``.

    Requires ``smac >= 2.0.0`` (``pip install pdmlabs[smac]``).
    """

    supports_categorical: bool = True  # ConfigSpace supports CategoricalHP natively

    # populated by _convert_param_space; used by _cast_ordinals
    _ordinal_types: dict

    # ------------------------------------------------------------------ #
    # Public API                                                           #
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
        """Maximize *objective_fn* using SMAC3.

        SMAC3 minimises internally; the objective is negated so SMAC
        effectively maximises. Results are negated back before returning.
        """
        return self._run(param_space, objective_fn, n_iterations, n_jobs,
                         initial_random=initial_random, negate=True)

    def minimize(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        constraint_fn=None,
    ) -> dict:
        """Minimise *objective_fn* using SMAC3 directly (no negation)."""
        return self._run(param_space, objective_fn, n_iterations, n_jobs,
                         initial_random=initial_random, negate=False)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        negate: bool,
    ) -> dict:
        """Common SMAC3 execution path used by both maximize and minimize.

        *n_iterations* is the **total** trial budget passed to SMAC's
        ``Scenario`` as ``n_trials``.  The first ``initial_random`` of those
        trials are random configurations drawn by the initial design
        (``HyperparameterOptimizationFacade.get_initial_design`` with
        ``n_configs=initial_random``); the remainder are Bayesian proposals.
        SMAC counts the initial design inside ``n_trials``, so no arithmetic
        adjustment is required.
        """
        try:
            from smac import HyperparameterOptimizationFacade, Scenario  # lazy import
            from smac.main.exceptions import ConfigurationSpaceExhaustedException
            from smac.runhistory.enumerations import StatusType
        except ImportError as exc:
            if "DTYPE" in str(exc):
                # SMAC <= 2.4.0 imports sklearn.tree._tree.DTYPE, dropped in
                # scikit-learn 1.9. Downgrading sklearn is not an option: the
                # scikit-survival core dependency pins it to >=1.9,<1.10.
                hint = (
                    "The installed SMAC is older than 2.4.1 and is incompatible "
                    "with scikit-learn >= 1.9. Upgrade it with: "
                    "pip install -U 'smac>=2.4.1'"
                )
            else:
                hint = "Install SMAC3 support with: pip install 'pdmlabs[smac]'"
            raise ImportError(f"Could not import SMAC ({exc}). {hint}") from exc

        configspace = self._convert_param_space(param_space)
        output_dir, keep_output = self._make_output_dir()

        cleanup = None
        if not keep_output:
            def cleanup():
                shutil.rmtree(output_dir, ignore_errors=True)

            # Safety net for interpreter exit paths that skip the finally below.
            atexit.register(cleanup)

        sign = -1.0 if negate else 1.0

        smac = None
        try:
            def smac_target(config, seed: int = 0) -> float:
                params = self._cast_ordinals(dict(config))
                return sign * float(objective_fn(**params))

            scenario = Scenario(
                configspace,
                deterministic=True,
                seed=self.random_state,
                n_trials=n_iterations,
                n_workers=n_jobs,
                output_directory=Path(output_dir),
            )

            # Build an explicit initial design so that exactly *initial_random*
            # random configurations are evaluated before BO kicks in.
            # n_configs overrides SMAC's default n_configs_per_hyperparameter heuristic.
            initial_design = HyperparameterOptimizationFacade.get_initial_design(
                scenario,
                n_configs=max(1, initial_random),
            )

            # SMAC's default 16 retries make it raise
            # ConfigurationSpaceExhaustedException once most of a small grid has
            # been tried (the random fallback keeps drawing seen configs).
            config_selector = HyperparameterOptimizationFacade.get_config_selector(
                scenario, retries=max(16, 10 * n_iterations),
            )
            smac = HyperparameterOptimizationFacade(
                scenario,
                smac_target,
                initial_design=initial_design,
                config_selector=config_selector,
                overwrite=True,
            )
            _deterministic_start_points(smac)
            try:
                incumbent = smac.optimize()
            except ConfigurationSpaceExhaustedException:
                # Keep what was evaluated instead of losing the whole search.
                _log.warning("SMAC3Adapter: no unseen configuration left; returning "
                             "the %d trials that ran.", smac.runhistory.finished)
                _drain_running_trials(smac)
                incumbent = None

            # Extract full run history
            params_tried, objective_values = [], []
            seen: set = set()
            for trial_key, trial_value in smac.runhistory.items():
                if trial_value.status != StatusType.SUCCESS:
                    continue  # crashed / never finished: there is no score
                cfg_id = trial_key.config_id
                if cfg_id in seen:
                    continue
                seen.add(cfg_id)
                config = smac.runhistory.get_config(cfg_id)
                raw_cost = trial_value.cost
                if isinstance(raw_cost, (list, tuple)):
                    raw_cost = raw_cost[0]
                params_tried.append(self._cast_ordinals(dict(config)))
                # re-apply sign to get back the original objective direction
                objective_values.append(sign * float(raw_cost))

            # Derive best from the history so the result is always consistent
            if objective_values:
                if negate:  # we were maximising
                    best_idx = objective_values.index(max(objective_values))
                else:
                    best_idx = objective_values.index(min(objective_values))
                best_params = params_tried[best_idx]
                best_objective = objective_values[best_idx]
            else:
                best_params = self._cast_ordinals(dict(incumbent)) if incumbent else {}
                best_objective = 0.0

        finally:
            if cleanup is not None:
                # Stop the dask workers before deleting the scratch space they
                # are rooted in, so the rmtree cannot race a live write.
                self._shutdown_dask(smac)
                cleanup()
                atexit.unregister(cleanup)

        return {
            "best_params": best_params,
            "best_objective": best_objective,
            "params_tried": params_tried,
            "objective_values": objective_values,
        }

    def _make_output_dir(self):
        """Create SMAC's scratch directory and say whether to keep it.

        ``mkdtemp`` uses ``O_CREAT | O_EXCL``, so the directory is unique against
        every process on the machine -- concurrent runs cannot collide. It is
        placed under ``$TMPDIR``, never the caller's working directory.

        Setting :data:`OUTPUT_DIR_ENV_VAR` places it under that base instead and
        retains it after the run, for inspecting SMAC's own JSON output.

        Returns
        -------
        tuple[str, bool]
            The absolute directory path, and whether to keep it.
        """
        base = os.environ.get(OUTPUT_DIR_ENV_VAR) or None
        keep = base is not None
        if base:
            try:
                os.makedirs(base, exist_ok=True)
            except OSError as exc:
                _log.warning(
                    "SMAC3Adapter: %s=%r is not usable (%s); falling back to "
                    "$TMPDIR and removing the directory after the run.",
                    OUTPUT_DIR_ENV_VAR, base, exc,
                )
                base = None
                keep = False

        output_dir = os.path.abspath(
            tempfile.mkdtemp(prefix="pdmlabs_smac_", dir=base)
        )
        return output_dir, keep

    @staticmethod
    def _shutdown_dask(smac) -> None:
        """Close SMAC's dask client, if it started one. Best effort.

        With ``n_workers > 1`` the facade wraps its runner in a
        ``DaskParallelRunner`` whose client is rooted *inside* the scenario's
        output directory (``dask-scratch-space/``, ``.dask_scheduler_file``).
        That runner closes itself in ``__del__``, but only whenever the garbage
        collector gets to it -- which may be after we have deleted the directory
        underneath it. Closing explicitly makes the ordering deterministic.

        ``close`` is defined only on ``DaskParallelRunner``, so the ``hasattr``
        guard makes this a no-op on the serial ``n_workers == 1`` path.

        Never raises: this is opportunistic cleanup running in a ``finally``, and
        must not turn a completed optimization into a failure.
        """
        if smac is None:
            return
        try:
            runner = getattr(smac, "_runner", None)
            if runner is not None and hasattr(runner, "close"):
                runner.close(force=True)
        except Exception as exc:  # noqa: BLE001 - deliberately swallowed
            _log.debug("SMAC3Adapter: could not close dask client (%s)", exc)

    def _convert_param_space(self, param_space: dict):
        """Translate PdMLabs param space dict to a ``ConfigurationSpace``."""
        from ConfigSpace import ConfigurationSpace  # lazy import
        from ConfigSpace.hyperparameters import (  # lazy import
            CategoricalHyperparameter,
            OrdinalHyperparameter,
            UniformFloatHyperparameter,
        )
        from scipy.stats._distn_infrastructure import rv_frozen

        cs = ConfigurationSpace()
        self._ordinal_types = {}  # reset for this call

        for name, values in param_space.items():
            if isinstance(values, rv_frozen):
                hp = UniformFloatHyperparameter(
                    name,
                    lower=float(values.ppf(0.01)),
                    upper=float(values.ppf(0.99)),
                )
                # rv_frozen is treated as continuous float; no ordinal casting needed
            elif isinstance(values, list):
                if any(isinstance(v, str) for v in values):
                    hp = CategoricalHyperparameter(name, choices=values)
                else:
                    # Numeric or bool — preserve exact candidates
                    if all(isinstance(v, bool) for v in values):
                        self._ordinal_types[name] = "bool"
                    elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                        self._ordinal_types[name] = "int"
                    else:
                        self._ordinal_types[name] = "float"
                    hp = OrdinalHyperparameter(
                        name, sequence=[str(v) for v in values]
                    )
            else:
                raise ValueError(
                    f"Unsupported param_space value type for '{name}': {type(values)}"
                )
            cs.add_hyperparameter(hp)

        return cs

    def _cast_ordinals(self, params: dict) -> dict:
        """Cast string ordinal values back to their original numeric types.

        ``OrdinalHyperparameter`` stores values as strings in ConfigSpace.
        This method converts them back using the type map built by
        ``_convert_param_space``.
        """
        result = {}
        for k, v in params.items():
            if k in self._ordinal_types:
                t = self._ordinal_types[k]
                if t == "bool":
                    result[k] = v == "True"
                elif t == "int":
                    result[k] = int(float(v))
                else:
                    result[k] = float(v)
            else:
                result[k] = v
        return result


def _drain_running_trials(smac) -> None:
    """Collect trials still running on dask workers after optimize() raised."""
    try:
        smbo = smac.optimizer
        while smbo._runner.is_running():
            smbo._runner.wait()
            smbo._add_results()
    except Exception as exc:  # noqa: BLE001 - best effort
        _log.debug("SMAC3Adapter: could not drain running trials (%s)", exc)


def _deterministic_start_points(smac) -> None:
    """Make SMAC's local-search start order independent of PYTHONHASHSEED.

    ``LocalSearch._get_init_points_from_previous_configs`` returns
    ``list(set(configs))``; ``Configuration.__hash__`` is ``hash(repr(self))``,
    a salted string hash, so the start order -- and with it the local search's
    RNG consumption and the proposals -- changed on every interpreter start.
    The set order was arbitrary anyway; sorting by repr makes it stable.
    Private SMAC API: a no-op if the attribute layout ever changes.
    """
    local_search = getattr(getattr(smac, "_acquisition_maximizer", None), "_local_search", None)
    original = getattr(local_search, "_get_init_points_from_previous_configs", None)
    if original is None:
        return

    def ordered(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=repr)

    local_search._get_init_points_from_previous_configs = ordered
