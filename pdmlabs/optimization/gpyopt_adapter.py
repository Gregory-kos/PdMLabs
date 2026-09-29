"""GPyOpt Bayesian Optimization adapter for PdMLabs.

All ``GPyOpt`` imports are **lazy** (inside method bodies) so this module is
importable even when ``gpyopt`` is not installed.

Install GPyOpt support with::

    pip install pdmlabs[gpyopt]

The adapter uses ``GPyOpt.methods.BayesianOptimization`` as the single
entry point. Hyperparameter space conversion rules:

* ``list`` with **string** values       → ``'categorical'`` domain
  (GPyOpt uses integer indices internally; cast back via original list)
* ``list`` with **numeric** values      → ``'discrete'`` domain
  (preserves the exact candidate set defined by the user)
* ``scipy.stats`` frozen dist           → ``'continuous'`` domain
  (bounds derived from ``ppf(0.01)`` / ``ppf(0.99)``)

Parallelism works as follows:

* ``batch_size=n_jobs`` tells the acquisition function to propose ``n_jobs``
  candidates per BO iteration (``'local_penalization'`` evaluator when
  ``n_jobs > 1``, ``'sequential'`` otherwise).
* Inside ``gpyopt_target``, ``joblib.Parallel(n_jobs=n_jobs)`` evaluates
  those ``n_jobs`` candidates concurrently — mirroring how Mango's
  ``@scheduler.parallel(n_jobs)`` evaluates a batch of configs in parallel.
* ``num_cores`` is always set to ``1`` to prevent GPyOpt from spawning an
  additional layer of subprocesses on top of ``joblib``.
* The number of BO iterations is set so that total function evaluations
  (``initial_random + n_iterations × n_jobs``) stay within the ``MAX_RUNS``
  budget — see ``calculate_optimizer_budget`` in ``utils.py``.
"""

import numpy as np
from pdmlabs.optimization.base import BaseOptimizerAdapter


class _BatchObjective:
    """Evaluates a whole GPyOpt batch with one call to *func* (in-process).

    Stands in for GPyOpt's ``SingleObjective``; the adapter's ``gpyopt_target``
    already runs the rows of a batch in parallel through joblib.
    """

    def __init__(self, func):
        self.func = func
        self.num_evaluations = 0

    def evaluate(self, x):
        y = self.func(x)
        self.num_evaluations += x.shape[0]
        return y, [0.0] * x.shape[0]


class GPyOptAdapter(BaseOptimizerAdapter):
    """Adapter for ``GPyOpt.methods.BayesianOptimization``.

    Requires ``gpyopt`` and ``GPy >= 1.0.8``
    (``pip install pdmlabs[gpyopt]``).

    Parallelism model
    -----------------
    ``batch_size=n_jobs`` candidates are proposed per BO iteration by the
    ``'local_penalization'`` acquisition evaluator (or ``'sequential'``
    when ``n_jobs == 1``).  The ``gpyopt_target`` wrapper evaluates those
    candidates in parallel via ``joblib.Parallel(n_jobs=n_jobs)``,
    matching Mango's ``@scheduler.parallel`` approach.  ``num_cores`` is
    always ``1`` so GPyOpt does not add a second layer of subprocesses.
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
        """Maximise *objective_fn* using GPyOpt Bayesian optimisation."""
        return self._run(
            param_space, objective_fn, n_iterations, n_jobs, initial_random,
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
        """Minimise *objective_fn* using GPyOpt Bayesian optimisation."""
        return self._run(
            param_space, objective_fn, n_iterations, n_jobs, initial_random,
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
        n_jobs: int,
        initial_random: int,
        negate: bool,
    ) -> dict:
        """Common GPyOpt execution path used by both maximize and minimize.

        Parameters
        ----------
        param_space:
            PdMLabs native hyperparameter search space dict.
        objective_fn:
            Raw ``(**params) -> float`` callable (no decorator required).
        n_iterations:
            Number of BO iterations passed to ``run_optimization``.
            Set by ``calculate_optimizer_budget`` so that
            ``initial_random + n_iterations × n_jobs ≤ MAX_RUNS``.
        n_jobs:
            Number of candidates proposed and evaluated in parallel per
            BO iteration.  Controls both ``batch_size`` (acquisition) and
            ``joblib.Parallel(n_jobs=…)`` (objective evaluation).
        initial_random:
            Warm-up evaluations before BO starts
            (``initial_design_numdata``).
        negate:
            If ``True`` the objective is negated before being passed to
            GPyOpt (which always minimises), enabling maximisation.

        Returns
        -------
        dict
            ``{'best_params', 'best_objective', 'params_tried',
            'objective_values'}`` — values are in the original (un-negated)
            direction of the caller.
        """
        import GPyOpt  # lazy import

        sign = -1.0 if negate else 1.0
        domain, param_names, type_map = self._build_domain(param_space)

        def gpyopt_target(X: np.ndarray) -> np.ndarray:
            # X shape: (n_points, n_params) — n_points == batch_size each call.
            # joblib.Parallel evaluates all n_points rows concurrently, mirroring
            # how Mango's @scheduler.parallel(n_jobs) evaluates a batch of configs.
            from joblib import Parallel, delayed  # lazy import

            def _eval_single(row: np.ndarray) -> list:
                params = self._row_to_params(row, param_names, type_map)
                return [sign * float(objective_fn(**params))]

            results = Parallel(n_jobs=n_jobs)(
                delayed(_eval_single)(row) for row in X
            )
            return np.array(results)  # shape: (n_points, 1)

        # Not local_penalization: its estimate_L does ``res.fun[0][0]`` while
        # scipy >= 1.14 returns a scalar from minimize(), so it raises TypeError on
        # every batch. thompson_sampling is model-aware, honours batch_size, and
        # keeps genuine parallel batching.
        evaluator_type = "thompson_sampling" if n_jobs > 1 else "sequential"

        # GPyOpt has no seed argument: its initial design and acquisition
        # optimiser both draw from NumPy's global RNG, so seeding that here is
        # the only hook available.
        np.random.seed(self.random_state)

        # Draw and evaluate the warm-up ourselves. Passing f= would let GPyOpt
        # build SingleObjective(f, batch_size) -- it hands batch_size over as
        # num_cores -- which forks one process per row, each opening its own
        # loky pool, so every batch idles until loky's 300 s worker timeout and
        # an exception in any trial deadlocks the parent on Pipe.recv().
        from GPyOpt.core.errors import FullyExploredOptimizationDomainError
        from GPyOpt.core.task.space import Design_space
        from GPyOpt.experiment_design import initial_design

        n_init = max(1, initial_random)
        candidates = initial_design("random", Design_space(domain), 20 * n_init)
        # Distinct warm-up rows where the grid allows it (random design samples
        # with replacement).
        unique_rows = list(dict.fromkeys(map(tuple, candidates)))[:n_init]
        X_init = np.array(unique_rows, dtype=float)
        Y_init = gpyopt_target(X_init)

        bo = GPyOpt.methods.BayesianOptimization(
            f=None,           # objective attached below: whole batch, one call
            domain=domain,
            X=X_init,
            Y=Y_init,
            evaluator_type=evaluator_type,
            batch_size=n_jobs,
            num_cores=1,
            maximize=False,   # GPyOpt always minimises; sign flip handled above
            verbosity=False,
            de_duplication=True,  # never re-propose an evaluated configuration
        )
        bo.objective = _BatchObjective(gpyopt_target)
        try:
            # eps=-1: GPyOpt otherwise stops as soon as two consecutive
            # evaluations coincide, which in a discrete space ends the run
            # after one or two batches.
            bo.run_optimization(max_iter=n_iterations, eps=-1)
        except FullyExploredOptimizationDomainError:
            bo._compute_results()  # every configuration has been evaluated

        best_params = self._row_to_params(
            bo.x_opt.flatten(), param_names, type_map
        )
        # bo.fx_opt stores sign × true_score (set inside gpyopt_target).
        # Multiplying by sign recovers true_score: sign × (sign × s) = s.
        best_objective = sign * float(bo.fx_opt)

        # Extract full evaluation history
        params_tried: list[dict] = []
        objective_values: list[float] = []
        for i in range(bo.X.shape[0]):
            p = self._row_to_params(bo.X[i], param_names, type_map)
            params_tried.append(p)
            objective_values.append(sign * float(bo.Y[i, 0]))

        return {
            "best_params": best_params,
            "best_objective": best_objective,
            "params_tried": params_tried,
            "objective_values": objective_values,
        }

    # ------------------------------------------------------------------ #
    # Domain construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_domain(self, param_space: dict):
        """Translate PdMLabs param_space to a GPyOpt domain list + metadata.

        Returns
        -------
        domain : list[dict]
            GPyOpt domain descriptor (one entry per hyperparameter).
        param_names : list[str]
            Ordered list of hyperparameter names (matches domain order).
        type_map : dict
            ``name → {'kind': str, 'values': list | None}`` used by
            ``_row_to_params`` to cast GPyOpt float64 outputs back to the
            correct Python type.
        """
        from scipy.stats._distn_infrastructure import rv_frozen  # lazy

        domain: list[dict] = []
        param_names: list[str] = []
        # kind ∈ {'int', 'float', 'str', 'continuous'}
        type_map: dict[str, dict] = {}

        for name, values in param_space.items():
            param_names.append(name)

            if isinstance(values, rv_frozen):
                domain.append({
                    "name": name,
                    "type": "continuous",
                    "domain": (float(values.ppf(0.01)), float(values.ppf(0.99))),
                })
                type_map[name] = {"kind": "continuous", "values": None}

            elif isinstance(values, list):
                if any(isinstance(v, str) for v in values):
                    # Categorical: pass integer indices to GPyOpt; map back
                    # via the original string list in _row_to_params.
                    domain.append({
                        "name": name,
                        "type": "categorical",
                        "domain": tuple(range(len(values))),
                    })
                    type_map[name] = {"kind": "str", "values": values}

                else:
                    # Numeric list: use 'discrete' to preserve exact candidates.
                    # GPyOpt constrains sampling to the provided tuple, so the
                    # returned float64 is always exactly one of these values.
                    domain.append({
                        "name": name,
                        "type": "discrete",
                        "domain": tuple(float(v) for v in values),
                    })
                    if all(isinstance(v, bool) for v in values):
                        type_map[name] = {"kind": "bool", "values": None}
                    elif all(isinstance(v, int) for v in values):
                        type_map[name] = {"kind": "int", "values": values}
                    else:
                        type_map[name] = {"kind": "float", "values": values}

            else:
                raise ValueError(
                    f"Unsupported param_space type for '{name}': {type(values)}"
                )

        return domain, param_names, type_map

    # ------------------------------------------------------------------ #
    # Type casting                                                         #
    # ------------------------------------------------------------------ #

    def _row_to_params(
        self,
        row: np.ndarray,
        param_names: list[str],
        type_map: dict,
    ) -> dict:
        """Cast a GPyOpt float64 row back to named params with original types.

        Notes
        -----
        * ``'discrete'`` bool params: rounded and cast back to bool (0.0->False, 1.0->True).
        * ``'discrete'`` int params: GPyOpt constrains to the provided tuple
          so the returned float is exactly the original integer value (e.g.
          ``100.0``).  ``int(float(val))`` is sufficient — no rounding needed.
        * ``'discrete'`` float params: snap to nearest candidate in the
          original list via ``min(..., key=...)``.
        * ``'categorical'`` (str) params: GPyOpt returns an integer index;
          look up the original string in ``type_map[name]['values']``.
        * ``'continuous'`` params: return the raw float unchanged.
        """
        params: dict = {}
        for i, name in enumerate(param_names):
            meta = type_map[name]
            raw = float(row[i])

            if meta["kind"] == "bool":
                params[name] = bool(round(raw))

            elif meta["kind"] == "int":
                params[name] = int(float(raw))

            elif meta["kind"] == "float":
                # Snap to nearest candidate in the original discrete list
                candidates = meta["values"]
                params[name] = min(candidates, key=lambda c: abs(c - raw))

            elif meta["kind"] == "str":
                idx = int(float(raw))
                idx = max(0, min(idx, len(meta["values"]) - 1))
                params[name] = meta["values"][idx]

            else:  # 'continuous'
                params[name] = raw

        return params
