"""Lifecycle energy study driver.

``run_experiment`` searches over a single dataset dict; it has no notion of a
held-out test split and performs no final pass. The established protocol lives
in the ``test/`` scripts: split with :class:`~pdmlabs.utils.dataset.Dataset`
(every ``get_*_dataset`` returns ``(train, test)``), search on the first half,
then score the second. This module formalises that and wraps energy envelopes
around each stage, producing the phase decomposition the analysis needs:

``E_search``     HPO: fit on the training episodes, scored on the VALIDATION
                 episodes (the targets of the first dict) -- all trials plus
                 optimizer overhead
``E_final``      Refit + score the winning configuration on the TEST dict
``E_baseline_*`` The untuned-defaults and random-search references, scored on
                 validation like SEARCH; each also gets a ``*_FINAL`` test pass
                 so tuned and baseline quality are compared on the same split

Two baselines, answering different questions
---------------------------------------------
*defaults* answers "was tuning worth the energy at all?"; *random search* at an
equal trial budget answers the sharper question for an HPO comparison, "was
*smart* tuning worth the energy over dumb tuning?". A method can easily win the
first and lose the second.

Protocol notes that are easy to get wrong
------------------------------------------
* ``use_cache`` is forced off. A cache hit does no work but still runs an
  O(n^2) ``mlflow.search_runs``, which collapses ``sum(E_trial)`` while
  ``E_search`` stays large -- indistinguishable from enormous optimizer
  overhead.
* The final pass uses ``INITIAL_RANDOM=0``. With ``MAX_RUNS=1`` and
  ``INITIAL_RANDOM=1``, ``validate_budget_feasibility`` computes a minimum of
  two evaluations and **raises**.
* Single-evaluation stages (the defaults baseline, and the final pass) are run
  with the ``mango`` backend regardless of the optimizer under test. With one
  configuration there is nothing to search, so the choice is immaterial to the
  result -- and it sidesteps a pre-existing defect where ``mango_random``
  raises ``KeyError: 'best_params'`` whenever ``n_iterations == 0``, which is
  precisely what ``calculate_optimizer_budget`` returns for ``MAX_RUNS=1``.
  The optimizer a final pass belongs to is recorded in its experiment name.
* Repeats are interleaved (repeat outer, optimizer inner) rather than run back
  to back, so thermal drift is spread across conditions instead of being
  confounded with whichever optimizer ran last.
* ``MAX_JOBS`` is forced to 1: concurrent trials share machine-wide energy
  counters and cannot be attributed individually.
* **Each repeat gets its own seed** (``base_seed + repeat_idx``) and every
  record is stamped with both. Repeats that share a seed rerun the identical
  search, so their spread measures hardware noise rather than the optimizer's
  run-to-run variance -- which is usually the quantity of interest.
* **BLAS thread counts are pinned** for the duration of the study. Left at
  their default, libraries use every core, and a method whose internals happen
  to parallelise more makes its *optimizer* look energy-hungry for reasons that
  have nothing to do with search.
* **Idle is measured before and after** each repeat. A baseline that drifts
  between the two means ambient conditions moved, and every ``dynamic_j``
  derived from it is suspect -- so the drift is recorded rather than averaged
  away.
"""

from __future__ import annotations

import logging
import os
import time

_log = logging.getLogger(__name__)


_BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class _PinnedThreads:
    """Pin BLAS thread counts for the study, restoring them afterwards.

    Set as environment variables because the numerical libraries read them at
    import time in each worker process; setting them in-process after import
    would be silently ignored by some of them.
    """

    def __init__(self, n_threads):
        self.n_threads = n_threads
        self._saved = {}

    def __enter__(self):
        if not self.n_threads:
            return self
        for v in _BLAS_VARS:
            self._saved[v] = os.environ.get(v)
            os.environ[v] = str(self.n_threads)
        _log.info("Pinned BLAS threads to %d for this study", self.n_threads)
        return self

    def __exit__(self, *exc):
        for v, old in self._saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
        return False


#: Backend used for stages that evaluate exactly one configuration. See the
#: module docstring: with nothing to search the choice cannot affect the result,
#: and ``mango_random`` is broken at ``n_iterations == 0``.
SINGLE_EVAL_OPTIMIZER = "mango"


def _collapse(best_params, prefix="method_"):
    """Turn a winning param dict into a one-point search space."""
    out = {}
    for k, v in (best_params or {}).items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = [v]
    return out


def _extra_params(best_params, prefix="method_"):
    """Non-method entries (preprocessor_/postprocessor_/thresholder_/profile_*)."""
    return {k: [v] for k, v in (best_params or {}).items() if not k.startswith(prefix)}


def run_energy_study(study_name, train_dataset, test_dataset, methods,
                     param_space_dict_per_method, method_names, experiments,
                     experiment_names, optimizers=("mango",), MAX_RUNS=20,
                     INITIAL_RANDOM=2, optimization_param="AD1_AUC",
                     maximize=True, energy_tracking="codecarbon",
                     energy_backend_kwargs=None, output_dir="./energy_runs",
                     n_repeats=1, measure_idle=True, idle_duration_s=60.0,
                     base_seed=42, blas_threads=4, bracket_idle=True,
                     include_default_baseline=True, include_random_baseline=True,
                     allow_estimated=False, run_final_pass=True,
                     **run_experiment_kwargs):
    """Run a full lifecycle energy study and export a tidy record set.

    Returns ``{'rows', 'summary', 'results', 'machine', 'idle', 'spool_dir'}``.

    Raises ``RuntimeError`` from the preflight when CPU energy would be a model
    rather than a measurement, unless *allow_estimated* is set -- see
    :func:`pdmlabs.energy.preflight`.
    """
    from pdmlabs.RunExperiment import run_experiment
    from pdmlabs.energy import preflight
    from pdmlabs.energy.machine import (save_machine_context,
                                        measure_idle as measure_idle_fn)
    from pdmlabs.energy.export import export_study

    bk = dict(energy_backend_kwargs or {})
    spool_dir = os.path.abspath(os.path.join(output_dir, study_name))
    os.makedirs(spool_dir, exist_ok=True)

    # 1. Refuse to start if "energy" would really mean "runtime".
    info = preflight(energy_tracking, allow_estimated=allow_estimated, **bk)
    _log.info("Energy preflight: cpu=%s gpu=%s ram=%s",
              info.get("power_source_cpu"), info.get("power_source_gpu"),
              info.get("ram_source"))

    # 2. Pin threads BEFORE fingerprinting, so machine.json records the values
    #    the study actually ran under rather than the ambient ones. The pin is
    #    held for the whole study and released in the finally below.
    pinned = _PinnedThreads(blas_threads)
    pinned.__enter__()

    # 3. Machine context and idle baseline, before any work warms the machine.
    machine, idle = save_machine_context(
        spool_dir, backend=energy_tracking, idle=measure_idle,
        idle_duration_s=idle_duration_s, backend_kwargs=bk)

    common = dict(methods=methods, method_names=method_names,
                  experiments=experiments, experiment_names=experiment_names,
                  optimization_param=optimization_param, maximize=maximize,
                  MAX_JOBS=1, energy_tracking=energy_tracking,
                  energy_output_dir=output_dir, energy_study_name=study_name,
                  energy_backend_kwargs=bk, **run_experiment_kwargs)
    common.pop("use_cache", None)          # forced off; see module docstring

    results = []
    idle_brackets = []

    def _search(tag, optimizer, spaces, runs, init_random, dataset, label,
                seed, rep, extra_common=None, extra_dims=None):
        t0 = time.time()
        kw = dict(common)
        if extra_common:
            kw.update(extra_common)
        dims = {"repeat_idx": rep, "seed": seed}
        # extra_dims may override 'optimizer'. The final pass runs on a fixed
        # single-eval backend, but its energy belongs to the optimizer whose
        # winner is being scored -- without the override every FINAL record
        # would be stamped with the single-eval backend, and the whole final
        # phase would be attributed to that one optimizer.
        if extra_dims:
            dims.update(extra_dims)
        best = run_experiment(dataset=dataset,
                              param_space_dict_per_method=spaces,
                              MAX_RUNS=runs, INITIAL_RANDOM=init_random,
                              optimizer=optimizer, use_cache=False,
                              energy_phase_label=label,
                              random_state=seed,
                              energy_extra_dims=dims,
                              **kw)
        return {"tag": tag, "optimizer": optimizer, "phase": label, "seed": seed,
                "repeat_idx": rep, "best": best, "wall_s": time.time() - t0}

    def _final(res, owner, label, seed, rep):
        """Refit *res*'s winner and score the TEST dict (a one-point space).

        Every stage except this one scores the first dict, whose targets are
        the VALIDATION episodes; tuned and baseline quality (and hence any
        payoff) are only comparable when all of them get this test pass.
        """
        if not (run_final_pass and test_dataset is not None):
            return None
        try:
            spaces, extras = [], {}
            for b in res["best"] or []:
                spaces.append(_collapse(b.get("best_params")))
                extras.update(_extra_params(b.get("best_params")))
            if not spaces:
                return None
            extra = {"experiment_names":
                     ["%s [final %s]" % (n, owner) for n in experiment_names]}
            if extras:
                extra["additional_parameters"] = extras
            fin = _search("final r%d" % rep, SINGLE_EVAL_OPTIMIZER, spaces, 1, 0,
                          test_dataset, label, seed, rep, extra_common=extra,
                          extra_dims={"optimizer": owner,
                                      "single_eval_backend": SINGLE_EVAL_OPTIMIZER})
            fin["optimizer"] = owner   # the winner this pass belongs to
            results.append(fin)
            return fin
        except Exception as exc:
            _log.warning("Final pass for '%s' failed: %s", owner, exc, exc_info=True)
            return None

    try:
        for rep in range(n_repeats):
            # A distinct seed per repeat: see the module docstring.
            seed = base_seed + rep

            idle_before = None
            if measure_idle and bracket_idle:
                try:
                    idle_before = measure_idle_fn(energy_tracking,
                                                  idle_duration_s,
                                                  backend_kwargs=bk)
                except Exception as exc:
                    _log.warning("Pre-repeat idle measurement failed: %s", exc)

            # 4. Baselines. Defaults first: it is the cheapest, and it is the
            #    reference every other number is read against.
            if include_default_baseline:
                try:
                    bd = _search(
                        "baseline_default r%d" % rep, SINGLE_EVAL_OPTIMIZER,
                        [{} for _ in methods], 1, 0, train_dataset,
                        "BASELINE_DEFAULT", seed, rep)
                    results.append(bd)
                    _final(bd, "baseline_default", "BASELINE_DEFAULT_FINAL", seed, rep)
                except Exception as exc:
                    _log.warning("Default baseline failed: %s", exc, exc_info=True)

            if include_random_baseline:
                try:
                    br = _search(
                        "baseline_random r%d" % rep, "mango_random",
                        param_space_dict_per_method, MAX_RUNS, INITIAL_RANDOM,
                        train_dataset, "BASELINE_RANDOM", seed, rep)
                    results.append(br)
                    _final(br, "baseline_random", "BASELINE_RANDOM_FINAL", seed, rep)
                except Exception as exc:
                    _log.warning("Random baseline failed: %s", exc, exc_info=True)

            # 5. The optimizers under test, interleaved across repeats.
            for opt in optimizers:
                try:
                    res = _search("search r%d" % rep, opt,
                                  param_space_dict_per_method, MAX_RUNS,
                                  INITIAL_RANDOM, train_dataset, "SEARCH",
                                  seed, rep)
                    results.append(res)
                except Exception as exc:
                    _log.warning("Search with optimizer '%s' failed: %s", opt,
                                 exc, exc_info=True)
                    continue

                # 6. Final pass: refit the winner and score the held-out split.
                #    A one-point space makes this exactly fit-then-predict with
                #    the winning configuration, under whichever flavor's own
                #    semantics apply -- no reimplementation of seven code paths.
                _final(res, opt, "FINAL", seed, rep)

            # Idle again after the repeat. A baseline that moved invalidates
            # every dynamic_j derived from it, so the drift is recorded rather
            # than smoothed away.
            if measure_idle and bracket_idle:
                try:
                    idle_after = measure_idle_fn(energy_tracking,
                                                 idle_duration_s,
                                                 backend_kwargs=bk)
                    a = (idle_before or {}).get("idle_w")
                    b = (idle_after or {}).get("idle_w")
                    drift = abs(b - a) / max(a, b) if (a and b) else None
                    idle_brackets.append({
                        "repeat_idx": rep, "before_w": a, "after_w": b,
                        "drift_frac": drift,
                        "stable": (drift is not None and drift <= 0.10)})
                    if drift is not None and drift > 0.10:
                        _log.warning(
                            "Idle baseline drifted %.1f%% across repeat %d "
                            "(tolerance 10%%). Idle-subtracted energy for this "
                            "repeat is unreliable.", drift * 100, rep)
                except Exception as exc:
                    _log.warning("Post-repeat idle measurement failed: %s", exc)

    finally:
        pinned.__exit__(None, None, None)

    # 7. Export. The spool is left in place: it is the raw data, and an export
    #    that might itself be wrong is not grounds for deleting it.
    rows, summary = export_study(spool_dir, study_id=study_name)
    summary["n_repeats"] = n_repeats
    summary["optimizers"] = list(optimizers)
    summary["idle_brackets"] = idle_brackets
    summary["blas_threads"] = blas_threads
    summary["base_seed"] = base_seed
    return {"rows": rows, "summary": summary, "results": results,
            "machine": machine, "idle": idle, "idle_brackets": idle_brackets,
            "spool_dir": spool_dir}


#: Backends that ignore ``n_jobs`` and always run trials sequentially. Including
#: them in a concurrency sweep produces flat lines that look like a finding.
SEQUENTIAL_BACKENDS = {"hyperopt"}


def run_concurrency_study(study_name, train_dataset, methods,
                          param_space_dict_per_method, method_names,
                          experiments, experiment_names, optimizers=("mango",),
                          n_jobs_levels=(1, 4, 8), MAX_RUNS=20,
                          INITIAL_RANDOM=2, optimization_param="AD1_AUC",
                          maximize=True, energy_tracking="codecarbon",
                          energy_backend_kwargs=None,
                          output_dir="./energy_runs", base_seed=42,
                          blas_threads=4, measure_idle=True,
                          idle_duration_s=60.0, allow_estimated=False,
                          **run_experiment_kwargs):
    """Energy and wall-clock against parallelism -- the E-LQO RQ3 analogue.

    Parallelism trades wall-clock for contention: more workers finish sooner but
    each is slower, and total energy can move either way. Answering "does
    running the search wider cost more energy overall?" needs the sweep.

    **Phase-level energy only, by construction.** With ``n_jobs > 1`` trials run
    concurrently in worker processes and share machine-wide counters, so no
    trial can be attributed its own energy -- the instrumentation degrades to
    wall-clock automatically there. Phase energy stays valid at any ``n_jobs``,
    and is all this comparison needs.

    Returns ``{'rows', 'summary', 'results', 'machine', 'idle', 'spool_dir'}``.
    """
    from pdmlabs.RunExperiment import run_experiment
    from pdmlabs.energy import preflight
    from pdmlabs.energy.machine import save_machine_context
    from pdmlabs.energy.export import export_study

    bk = dict(energy_backend_kwargs or {})
    spool_dir = os.path.abspath(os.path.join(output_dir, study_name))
    os.makedirs(spool_dir, exist_ok=True)

    preflight(energy_tracking, allow_estimated=allow_estimated, **bk)

    pinned = _PinnedThreads(blas_threads)
    pinned.__enter__()
    machine, idle = save_machine_context(
        spool_dir, backend=energy_tracking, idle=measure_idle,
        idle_duration_s=idle_duration_s, backend_kwargs=bk)

    results = []
    try:
        # n_jobs outer, optimizer inner: interleaving conditions keeps thermal
        # drift from lining up with any single level of parallelism.
        for n_jobs in n_jobs_levels:
            for opt in optimizers:
                if opt in SEQUENTIAL_BACKENDS and n_jobs > 1:
                    _log.warning(
                        "Optimizer '%s' ignores n_jobs and runs sequentially; "
                        "skipping its n_jobs=%d cell rather than recording a "
                        "flat line that would read as a result.", opt, n_jobs)
                    continue
                try:
                    t0 = time.time()
                    best = run_experiment(
                        dataset=train_dataset,
                        param_space_dict_per_method=param_space_dict_per_method,
                        methods=methods, method_names=method_names,
                        experiments=experiments,
                        experiment_names=["%s [j%d]" % (n, n_jobs)
                                          for n in experiment_names],
                        MAX_RUNS=MAX_RUNS, MAX_JOBS=n_jobs,
                        INITIAL_RANDOM=INITIAL_RANDOM,
                        optimization_param=optimization_param,
                        maximize=maximize, optimizer=opt, use_cache=False,
                        random_state=base_seed,
                        energy_tracking=energy_tracking,
                        energy_output_dir=output_dir,
                        energy_study_name=study_name,
                        energy_backend_kwargs=bk,
                        energy_phase_label="SEARCH",
                        energy_extra_dims={"n_jobs_level": n_jobs,
                                           "seed": base_seed},
                        **run_experiment_kwargs)
                    results.append({"optimizer": opt, "n_jobs": n_jobs,
                                    "best": best, "wall_s": time.time() - t0})
                except Exception as exc:
                    _log.warning("Concurrency cell (%s, n_jobs=%d) failed: %s",
                                 opt, n_jobs, exc, exc_info=True)
    finally:
        pinned.__exit__(None, None, None)

    rows, summary = export_study(spool_dir, study_id=study_name)
    summary["n_jobs_levels"] = list(n_jobs_levels)
    summary["concurrency_cells"] = [
        {"optimizer": r["optimizer"], "n_jobs": r["n_jobs"], "wall_s": r["wall_s"]}
        for r in results]
    return {"rows": rows, "summary": summary, "results": results,
            "machine": machine, "idle": idle, "spool_dir": spool_dir}
