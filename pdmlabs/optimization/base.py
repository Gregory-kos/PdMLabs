"""Abstract base class for all PdMLabs optimizer adapters."""

import abc
from pdmlabs.exceptions.exception import CategoricalSpaceNotSupportedException


class BaseOptimizerAdapter(abc.ABC):
    """Abstract base for optimizer adapters.

    Subclasses translate the PdMLabs native param space (dict of
    str -> list | rv_frozen) into their backend's representation and
    expose a unified maximize / minimize API.

    The raw objective function passed to adapters has the signature::

        def optimization_objective(**params) -> float

    Each adapter is responsible for wrapping it into whatever calling
    convention its backend requires (batch list, single config, etc.).

    A single float is the whole contract: anything else a trial produces (its
    threshold, its fitted pipeline) travels back to the main process through
    :class:`pdmlabs.optimization.trial_sink.TrialSink`, which the objective
    closure carries into the workers. Adapters neither see nor forward it, so
    they need no changes to support it.
    """

    supports_categorical: bool = True  # set False in adapters that reject string params

    def __init__(self, random_state: int = 42):
        """Record the seed the backend should search with.

        Every adapter used to hardcode its own seed (or leave the backend on its
        default), which made ``Experiment.random_state`` inert: changing it
        varied the methods' RNGs but replayed the identical sequence of
        proposed configurations. Holding the seed here lets each backend feed
        it to whatever its own seeding hook is.

        Args:
            random_state: Seed for the backend's proposal RNG.
        """
        self.random_state = int(random_state)

    @abc.abstractmethod
    def maximize(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        constraint_fn=None,
    ) -> dict:
        """Run optimization maximizing *objective_fn* over *param_space*.

        Parameters
        ----------
        param_space:
            PdMLabs native space — ``str`` -> ``list`` or ``rv_frozen``.
        objective_fn:
            Raw callable ``(**params) -> float`` (single config, single score).
            Adapters wrap this into whatever their backend expects.
        n_iterations:
            Number of optimizer iterations / trials.
        n_jobs:
            Degree of parallelism (adapter-specific meaning).
        initial_random:
            Number of random warm-up evaluations (Mango only; ignored by SMAC).
        constraint_fn:
            Optional constraint predicate on parameter dicts (Mango only).

        Returns
        -------
        dict
            Keys: ``best_params`` (dict), ``best_objective`` (float),
            ``params_tried`` (list[dict]), ``objective_values`` (list[float]).
        """
        ...

    def minimize(
        self,
        param_space: dict,
        objective_fn,
        n_iterations: int,
        n_jobs: int,
        initial_random: int,
        constraint_fn=None,
    ) -> dict:
        """Minimize by negating *objective_fn* and delegating to maximize."""
        def negated(**params):
            return -objective_fn(**params)

        result = self.maximize(
            param_space, negated, n_iterations, n_jobs, initial_random, constraint_fn
        )
        result["best_objective"] = -result["best_objective"]
        result["objective_values"] = [-v for v in result["objective_values"]]
        return result

    def _check_categorical(self, param_space: dict) -> None:
        """Raise if this adapter cannot handle string-valued hyperparameters.

        Checks every entry in *param_space*. If ``self.supports_categorical``
        is ``False`` and any list value contains a ``str``, raises
        :class:`CategoricalSpaceNotSupportedException` immediately — before
        any trial is submitted — so the user receives a clear, actionable
        error.
        """
        if self.supports_categorical:
            return
        for param_name, values in param_space.items():
            if isinstance(values, list) and any(isinstance(v, str) for v in values):
                raise CategoricalSpaceNotSupportedException(
                    f"Optimizer '{type(self).__name__}' does not support categorical "
                    f"hyperparameter '{param_name}' (contains string values: {values})."
                )
