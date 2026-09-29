"""Optuna 5 TPE optimizer adapter for PdMLabs.

All optuna imports are lazy so this module is importable even when
optuna is not installed. Install with::

    pip install pdmlabs[optuna]          # installs optuna>=5.0.0

The adapter uses optuna.create_study + study.optimize as the single
entry point, with:

* TPESampler(seed=random_state, multivariate=True, constant_liar=True) --
  multivariate mode models joint distributions over all hyperparameters;
  constant_liar enables meaningful multi-process coordination by treating
  in-flight trials as if they returned the current best value.
  (Both are the new defaults in Optuna 5.0, but specified explicitly here
  for clarity and forward compatibility.)

* JournalStorage(JournalFileBackend) -- a file-backed log that allows
  multiple independent worker processes to share a single study without
  an RDB server.

Parallelism model (mirrors GPyOpt adapter)
------------------------------------------
joblib.Parallel(n_jobs=n_jobs, backend="loky") spawns n_jobs worker
processes. Each process connects to the shared JournalStorage file
and calls study.optimize(n_trials=trials_per_worker). The file uses
OS-level file locks, so concurrent writes are safe on a single machine.

Space conversion rules (all preserve exact candidate sets for lists):
  * list[int | float | bool | str | mixed] -> suggest_categorical(name, values)
  * rv_frozen                              -> suggest_float(name, ppf(0.01), ppf(0.99))

Serialization note
------------------
joblib's loky backend uses cloudpickle (not stdlib pickle), so closures
that capture ``self`` -- such as the ``optimization_objective`` defined
inside ``execute()`` -- are serializable without any special handling.

Writes to ``self.extra_metrics`` or ``self.best_pipeline`` inside a worker
subprocess update a deserialized copy of ``self`` that is discarded when the
process exits. Experiments therefore report per-trial artifacts through
:class:`pdmlabs.optimization.trial_sink.TrialSink` -- a filesystem channel the
objective closure carries into the workers -- rather than by assigning to
``self``. Adapter correctness is unaffected either way: the return dict
(``best_params``, ``best_objective``) is built from the shared JournalStorage
read back in the main process after all workers finish.
"""

import logging

import os
import tempfile

from pdmlabs.optimization.base import BaseOptimizerAdapter

_log = logging.getLogger(__name__)

# Study name prefix -- unique per PID to avoid collisions if called
# concurrently from multiple experiment instances in the same process.
_STUDY_PREFIX = "pdmlabs_optuna"


def _run_worker(
    worker_id: int,
    file_path: str,
    study_name: str,
    direction: str,
    n_trials: int,
    initial_random: int,
    param_space_items: list,
    objective_fn,
    random_state: int = 42,
) -> None:
    """Worker function executed inside each joblib subprocess.

    Must be a module-level function (not a closure or lambda) so that
    joblib's loky backend can pickle and spawn it into a subprocess.

    The optuna_objective wrapper is rebuilt here from ``param_space_items``
    and ``objective_fn`` rather than being passed pre-built, which avoids
    serializing the closure across process boundaries.

    Parameters
    ----------
    worker_id:
        Index of this worker (0-based; used only for debug logging).
    file_path:
        Absolute path to the shared JournalStorage log file.
    study_name:
        Name of the study to load (created before workers are spawned).
    direction:
        ``'maximize'`` or ``'minimize'``.
    n_trials:
        Number of trials this worker should run.
    initial_random:
        Passed to the worker's TPESampler to ensure correct warm-up sizing.
    param_space_items:
        ``list(param_space.items())`` -- the serialisable representation
        of the PdMLabs search space.
    objective_fn:
        Raw ``(**params) -> float`` callable from the experiment.
        Serialized by cloudpickle (loky's default); closures capturing
        ``self`` are handled transparently.
    """
    import optuna                                                      # lazy
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from scipy.stats._distn_infrastructure import rv_frozen           # lazy

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = JournalStorage(JournalFileBackend(file_path))
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(
            # Offset by worker_id: with one shared seed every worker would draw
            # the same warm-up configurations and waste n_jobs - 1 of them.
            seed=int(random_state) + worker_id,
            multivariate=True, constant_liar=True,
            n_startup_trials=max(1, initial_random),
        ),
    )

    def optuna_objective(trial) -> float:
        params = {}
        for name, values in param_space_items:
            if isinstance(values, rv_frozen):
                params[name] = trial.suggest_float(
                    name,
                    float(values.ppf(0.01)),
                    float(values.ppf(0.99)),
                )
            elif isinstance(values, list):
                # suggest_categorical preserves exact candidate sets for all
                # list types: int, float, bool, str, or mixed.
                params[name] = trial.suggest_categorical(name, values)
            else:
                raise ValueError(
                    f"Unsupported param_space type for '{name}': {type(values)}"
                )
        return float(objective_fn(**params))

    study.optimize(optuna_objective, n_trials=n_trials)


class OptunaAdapter(BaseOptimizerAdapter):
    """Adapter for Optuna 5 TPE (optuna>=5.0.0).

    Uses ``TPESampler(seed=self.random_state, multivariate=True, constant_liar=True)``
    (both now the defaults in Optuna 5.0) and a ``JournalStorage`` file
    created on the fly for GIL-free multi-process parallelism via
    ``joblib.Parallel(backend='loky')``.

    Requires ``optuna >= 5.0.0`` (``pip install pdmlabs[optuna]``).

    Parallelism
    -----------
    ``n_jobs`` independent worker processes each run
    ``floor(n_iterations / n_jobs)`` trials, sharing a single study via a
    temporary ``JournalStorage`` log file (file-locked, safe on a single
    machine). The temp file is deleted after the run.

    ``constant_liar=True`` instructs the TPE sampler to treat in-flight
    (not yet complete) trials as if they returned the current best value,
    enabling diverse candidate proposals across concurrent workers.
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
        """Maximise *objective_fn* using Optuna TPE."""
        return self._run(
            param_space, objective_fn, n_iterations, n_jobs,
            initial_random, direction="maximize",
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
        """Minimise *objective_fn* using Optuna TPE."""
        return self._run(
            param_space, objective_fn, n_iterations, n_jobs,
            initial_random, direction="minimize",
        )

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
        direction: str,
    ) -> dict:
        """Common Optuna execution path used by both maximize and minimize.

        Parameters
        ----------
        param_space:
            PdMLabs native hyperparameter search space dict.
        objective_fn:
            Raw ``(**params) -> float`` callable. Serialized by cloudpickle
            (loky's default); closures capturing ``self`` are fine.
        n_iterations:
            Total number of trials to run across all workers.
        n_jobs:
            Number of parallel worker processes (``loky`` backend).
        initial_random:
            Passed to ``TPESampler`` as ``n_startup_trials``. The first
            ``initial_random`` trials across the entire study are purely
            random before the TPE density estimator activates.
        direction:
            ``'maximize'`` or ``'minimize'`` -- passed to ``create_study``.

        Returns
        -------
        dict
            ``{'best_params', 'best_objective', 'params_tried',
            'objective_values'}`` -- in the original direction, built
            from the shared JournalStorage in the main process.

        Notes
        -----
        Total evaluations = ``n_iterations``. The adapter perfectly distributes
        these trials across the ``n_jobs`` workers. ``initial_random`` is passed
        as ``n_startup_trials`` and counts within this total budget.
        """
        import optuna                                                  # lazy
        from optuna.samplers import TPESampler                        # lazy
        from optuna.storages import JournalStorage                    # lazy
        from optuna.storages.journal import JournalFileBackend        # lazy
        from optuna.trial import TrialState                           # lazy
        from joblib import Parallel, delayed                          # lazy

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_name = f"{_STUDY_PREFIX}_{os.getpid()}"

        # Create shared journal file on-the-fly; deleted in the finally block.
        fd, tmp_file = tempfile.mkstemp(suffix=".log", prefix="pdmlabs_optuna_")
        os.close(fd)

        try:
            # Create the study once in the main process before spawning
            # workers. Workers call optuna.load_study to attach to it.
            storage = JournalStorage(JournalFileBackend(tmp_file))
            optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=direction,
                sampler=TPESampler(
                    seed=self.random_state,
                    multivariate=True,   # joint distribution over all params
                    constant_liar=True,  # treat in-flight trials for multi-process
                    n_startup_trials=max(1, initial_random),
                ),
            )

            # Spawn n_jobs worker processes. Each connects to the shared
            # JournalStorage file. Trials are distributed perfectly evenly.
            param_space_items = list(param_space.items())
            
            n_jobs = max(1, n_jobs)
            base_trials = n_iterations // n_jobs
            remainder = n_iterations % n_jobs
            
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_run_worker)(
                    i,
                    tmp_file,
                    study_name,
                    direction,
                    base_trials + (1 if i < remainder else 0),
                    initial_random,
                    param_space_items,
                    objective_fn,
                    self.random_state,
                )
                for i in range(n_jobs)
            )

            # Reload the study in the main process to read all results.
            storage = JournalStorage(JournalFileBackend(tmp_file))
            study = optuna.load_study(
                study_name=study_name, storage=storage
            )

            best_params = study.best_params
            best_objective = study.best_value

            # Extract full trial history.
            params_tried: list[dict] = []
            objective_values: list[float] = []
            for t in study.trials:
                if t.state == TrialState.COMPLETE:
                    params_tried.append(dict(t.params))
                    objective_values.append(t.value)

        finally:
            # Always clean up the temporary journal file.
            try:
                os.unlink(tmp_file)
            except OSError:
                pass

        return {
            "best_params": best_params,
            "best_objective": best_objective,
            "params_tried": params_tried,
            "objective_values": objective_values,
        }
