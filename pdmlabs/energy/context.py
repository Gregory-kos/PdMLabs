"""Per-process trial context, used to join an energy record to its MLflow run.

Why this exists
---------------
A decorator wrapping the trial objective **cannot** reach the trial's MLflow run
by proximity. In all seven experiment flavors the objective's ``return`` is
dedented *out* of the ``with mlflow.start_run(...)`` block (for example
``semi_supervised_experiment.py`` opens the run at line 74 and returns at line
222), so by the time the decorator's ``finally`` executes,
``mlflow.active_run()`` is already ``None``.

The join is therefore made explicitly, in two directions:

* energy record -> MLflow: :func:`bind_mlflow_run` is called as the **first**
  statement of ``PdMExperiment._finish_run``, which is the per-trial teardown
  funnel every flavor uses, runs in the same process as the objective, and still
  has the run active. Binding *first* -- ahead of ``log_model`` and friends,
  which can raise -- means the join key survives a partial teardown.
* MLflow -> energy record: the same call site logs ``energy_trial_id`` as an
  MLflow param.

Both keys are recorded; ``params_hash`` is the fallback when a trial opened no
run at all (the cached short-circuit).

Same invariant as :mod:`pdmlabs.optimization.trial_sink`: this is a module
global of an *importable* module, so it is never pickled by value, and it is
keyed by ``os.getpid()`` so a forked child cannot overwrite its parent's
in-flight trial.
"""

from __future__ import annotations

import os
import threading
import uuid

_STATE: dict = {}
_LOCK = threading.Lock()


def begin_trial(params_hash=None):
    """Open a trial context for this process and return its id."""
    tid = uuid.uuid4().hex
    with _LOCK:
        _STATE[os.getpid()] = {
            "energy_trial_id": tid,
            "params_hash": params_hash,
            "mlflow_run_id": None,
            "mlflow_experiment_id": None,
            "cached": False,
        }
    return tid


def bind_mlflow_run(run):
    """Attach the active MLflow run to this process's trial context.

    Returns the ``energy_trial_id`` so the caller can log it as a param, or
    ``None`` when there is no context or no run. Never raises: it sits ahead of
    the rest of ``_finish_run`` and must not be able to break trial teardown.
    """
    try:
        if run is None:
            return None
        with _LOCK:
            st = _STATE.get(os.getpid())
            if st is None:
                return None
            st["mlflow_run_id"] = run.info.run_id
            st["mlflow_experiment_id"] = run.info.experiment_id
            return st["energy_trial_id"]
    except Exception:
        return None


def mark_cached():
    """Flag the in-flight trial as served from cache.

    A cached trial returns before ``mlflow.start_run``, so it is not observable
    from outside the objective. It must be excluded from per-trial energy
    distributions, or cache hits drag the mean toward zero.
    """
    try:
        with _LOCK:
            st = _STATE.get(os.getpid())
            if st is not None:
                st["cached"] = True
    except Exception:
        pass


def current():
    """Read the in-flight context without consuming it."""
    with _LOCK:
        st = _STATE.get(os.getpid())
        return dict(st) if st else {}


def end_trial():
    """Pop and return this process's trial context."""
    with _LOCK:
        return _STATE.pop(os.getpid(), None) or {}


# --------------------------------------------------------------------------- #
# Active backend, per process.
#
# The trial decorator must record against the SAME backend instance that has the
# enclosing phase open, otherwise ``sum(task energy) <= phase energy`` stops
# holding and ``E_optimizer_overhead`` becomes a difference of two unrelated
# integrals.
#
# Keying by pid gives the n_jobs>1 safety property for free: the experiment sets
# the backend in the MAIN process only, so a loky/dask worker looks up its own
# pid, finds nothing, and records wall-clock only. Concurrent trials therefore
# cannot each be attributed the whole machine's energy -- the failure mode that
# would inflate every per-trial number by roughly n_jobs while still looking
# entirely plausible.
# --------------------------------------------------------------------------- #

_BACKENDS: dict = {}


def set_active_backend(backend):
    """Register *backend* as this process's measuring backend (None clears)."""
    with _LOCK:
        if backend is None:
            _BACKENDS.pop(os.getpid(), None)
        else:
            _BACKENDS[os.getpid()] = backend


def get_active_backend():
    """Return this process's active backend, or ``None`` in a worker process."""
    with _LOCK:
        return _BACKENDS.get(os.getpid())
