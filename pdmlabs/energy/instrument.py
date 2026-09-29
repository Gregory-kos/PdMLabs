"""Per-trial energy instrumentation for the HPO objective.

One decorator, applied at the single point in
``PdMExperiment._run_optimizer`` where the objective is handed to the adapter,
covers every one of the six optimizer backends -- each of them ultimately calls
the same ``objective_fn(**params)``.

Hard constraints, each learned from the existing code
------------------------------------------------------
* **Plain nested function, not a class or ``functools.partial``.** Mango assigns
  attributes to the objective (``mango/scheduler.py`` sets ``func.is_wrapped``
  and ``func.batch_size``; ``mango/tuner.py`` reads ``objective.batch_size`` to
  size its batches). A ``__slots__`` instance or a ``partial`` would break Mango
  outright.
* **The return value passes through untouched.** Adapters do
  ``float(objective_fn(**params))``.
* **Exceptions propagate untouched.** Every flavor does ``if self.debug: raise e``
  and ``debug=True`` is the default, so swallowing here would silently change
  HPO behaviour.
* **Nothing in ``finally`` may raise**, or a measurement failure would abort a
  search.
* **Picklable**, since the closure ships to loky/dask workers when ``n_jobs>1``.
  It carries only an :class:`~pdmlabs.energy.sink.EnergySink` (strings only) and
  plain dicts -- never a tracker, file handle or lambda.

Four exit paths, not three
--------------------------
1. normal ``return best_metrics_dict[...]`` -> ``exit_path="normal"``
2. the flavor's own handled-exception path, which returns ``0`` in six flavors
   but ``1`` in ``SA_experiment.py``. From here this is **indistinguishable from
   a normal return** -- the flavor catches the exception internally and returns
   a sentinel -- so it is recorded as ``exit_path="normal"`` with that sentinel
   as ``objective_value``. Identifying it after the fact means joining on the
   flavor's own MLflow record via ``mlflow_run_id``. The energy is still charged
   either way, which is the point: a failed trial burns real energy, and E-LQO
   likewise charges timed-out exploratory plans up to termination.
3. the ``_check_cached_run`` short-circuit -> ``exit_path="cached"``, so cache
   hits are excluded from per-trial distributions rather than dragging the mean
   toward zero
4. an exception that escapes entirely (``debug=True``, the default) ->
   ``exit_path="error"``, re-raised unchanged
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time

_log = logging.getLogger(__name__)


def _hash_params(params_norm):
    """Stable short hash of a normalised param dict, for the fallback join key."""
    try:
        blob = json.dumps(params_norm, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def measured_trial(objective_fn, sink, phase="SEARCH", dims=None):
    """Wrap a raw ``(**params) -> float`` objective with per-trial energy accounting.

    The backend is looked up per process via
    :func:`pdmlabs.energy.context.get_active_backend`. In a worker process there
    is none, so the trial is recorded with wall-clock only and
    ``attribution="phase_only"`` -- which is the correct and safe behaviour,
    since concurrent trials share machine-wide counters and cannot be
    attributed individually.
    """
    dims = dict(dims or {})

    @functools.wraps(objective_fn)
    def wrapper(**params):
        from pdmlabs.energy import context
        try:
            from pdmlabs.optimization.trial_sink import normalise_params
            params_norm = normalise_params(params)
        except Exception:
            params_norm = {k: str(v) for k, v in params.items()}

        trial_id = context.begin_trial(params_hash=_hash_params(params_norm))
        backend = context.get_active_backend()

        token = None
        if backend is not None:
            try:
                token = backend.start_task(trial_id)
            except Exception as exc:
                _log.warning("energy: start_task failed (%s); trial recorded "
                             "without energy", exc)

        t0_unix, t0 = time.time(), time.perf_counter()
        value = None
        error = None
        try:
            value = objective_fn(**params)
            return value
        except BaseException as exc:                       # path 4
            error = "%s: %s" % (type(exc).__name__, exc)
            raise
        finally:
            duration = time.perf_counter() - t0
            reading = None
            if backend is not None:
                try:
                    reading = backend.stop_task(token)
                except Exception as exc:
                    _log.warning("energy: stop_task failed: %s", exc)
            try:
                ctx = context.end_trial()
                exit_path = ("error" if error is not None
                             else "cached" if ctx.get("cached")
                             else "normal")
                rec = dict(
                    kind="trial", phase=phase,
                    energy_trial_id=trial_id,
                    mlflow_run_id=ctx.get("mlflow_run_id"),
                    mlflow_experiment_id=ctx.get("mlflow_experiment_id"),
                    params_hash=ctx.get("params_hash"),
                    params=params_norm,
                    objective_value=value,
                    exit_path=exit_path,
                    cached=bool(ctx.get("cached")),
                    error=error,
                    t_start_unix=t0_unix,
                    duration_s=duration,
                )
                if reading is not None:
                    rec.update(
                        cpu_j=reading.cpu_j, gpu_j=reading.gpu_j,
                        ram_j=reading.ram_j, total_j=reading.total_j,
                        co2e_g=reading.co2e_g,
                        measured_duration_s=reading.duration_s,
                        power_source_cpu=reading.power_source_cpu,
                        power_source_gpu=reading.power_source_gpu,
                        ram_source=reading.ram_source,
                        backend=reading.backend,
                        quality=reading.quality,
                        is_measured=reading.is_measured,
                        carbon_intensity_g_per_kwh=reading.carbon_intensity_g_per_kwh,
                        attribution="exact",
                    )
                else:
                    rec.update(attribution="phase_only", quality="failed",
                               is_measured=False, backend="none")
                rec.update(dims)
                sink.record(**rec)
            except Exception as exc:
                _log.warning("energy: could not spool trial record: %s", exc)

    return wrapper
