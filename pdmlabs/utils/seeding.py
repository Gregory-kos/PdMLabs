"""Central RNG seeding and determinism switches.

One place seeds every RNG PdMLabs can reach -- Python's ``random``, NumPy's
legacy global ``RandomState``, and torch (CPU *and* every CUDA device, which
``torch.manual_seed`` covers on its own) -- and flips the CUDA / cuDNN knobs
that otherwise make a GPU run differ from itself.

Why a module rather than a few lines in :class:`~pdmlabs.experiment.experiment.Experiment`
-------------------------------------------------------------------------------------
Seeding has to happen in more than one process. ``Experiment.__init__`` runs in
the main process, but at ``n_jobs > 1`` every trial executes inside a loky or
dask worker. Under the ``spawn`` start method those workers start from a fresh
interpreter with unseeded RNGs, so a seed set in the parent never reaches them.
:func:`seed_for_params` gives each trial a seed derived from *its own
parameters*, so a configuration produces the same score no matter which worker
picks it up or in what order -- which is the property that makes an HPO run
reproducible, not merely deterministic.

Determinism caveats
-------------------
* ``PYTHONHASHSEED`` cannot be changed after the interpreter starts. Setting it
  here only reaches child processes; to pin hash randomisation in the parent
  too, export it in the shell before launching Python. This is not academic:
  the SMAC backend iterates containers whose order depends on string hashing,
  so with hash randomisation live its Bayesian phase proposes a different
  sequence on every run *even at a fixed seed*. :func:`set_global_seed` warns
  once when it finds the variable unset.
* ``CUBLAS_WORKSPACE_CONFIG`` is read when the CUDA context is created. If CUDA
  is already initialised when :func:`set_global_seed` runs, the setting is
  ignored by cuBLAS and a warning is emitted.
* ``torch.use_deterministic_algorithms`` is enabled with ``warn_only=True``: a
  handful of ops have no deterministic kernel, and hard-failing would take out
  methods that are otherwise fine.
"""

import hashlib
import logging
import os
import random

import numpy as np


DEFAULT_RANDOM_STATE = 42

_WARNED_HASHSEED = False


def set_global_seed(seed: int = DEFAULT_RANDOM_STATE, deterministic: bool = True) -> int:
    """Seed every global RNG and, optionally, disable non-deterministic CUDA paths.

    Args:
        seed: Seed applied to ``random``, ``numpy.random`` and torch.
        deterministic: When True, also turn off cuDNN benchmarking/autotuning
            and request deterministic algorithms. Set False to trade
            reproducibility for the cuDNN autotuner's speed.

    Returns:
        The seed that was applied, so callers can log or record it.
    """
    seed = int(seed)

    # Only reaches child processes -- see the module docstring.
    if 'PYTHONHASHSEED' not in os.environ:
        global _WARNED_HASHSEED
        if not _WARNED_HASHSEED:
            _WARNED_HASHSEED = True
            logging.warning(
                'PYTHONHASHSEED is not set, so hash randomisation is active and '
                'this run is not fully reproducible (the SMAC backend is '
                'affected in particular). It cannot be fixed from inside a '
                'running interpreter -- export PYTHONHASHSEED=%d before '
                'launching Python.', seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # inherited by worker processes

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    # Seeds the CPU generator and, on its own, every CUDA device generator.
    torch.manual_seed(seed)

    if not deterministic:
        return seed

    # The autotuner picks a convolution algorithm from timings, so the same
    # graph can take different code paths -- and give different numbers -- on
    # two runs of the same machine.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            logging.warning(
                'CUDA was already initialised before set_global_seed(), so '
                'CUBLAS_WORKSPACE_CONFIG cannot take effect. Export '
                'CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python for '
                'fully deterministic cuBLAS reductions.')
    else:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # older torch, or a backend that refuses
        logging.warning('Could not enable deterministic torch algorithms: %s', exc)

    return seed


def seed_for_params(base_seed: int, params: dict) -> int:
    """Derive a stable per-trial seed from *base_seed* and a parameter dict.

    The digest is taken with blake2b rather than :func:`hash` because Python's
    string hashing is salted per interpreter: a ``hash()``-derived seed would
    differ between the parent and every worker process, which is exactly the
    situation this function exists to fix.

    Args:
        base_seed: The experiment's ``random_state``.
        params: The trial's parameter dict. Key order does not matter.

    Returns:
        A seed in ``[0, 2**31)``, identical for identical ``(base_seed, params)``.
    """
    payload = repr(sorted((str(k), repr(v)) for k, v in params.items()))
    digest = hashlib.blake2b(payload.encode('utf-8'), digest_size=8).digest()
    return (int(base_seed) + int.from_bytes(digest, 'big')) % (2 ** 31)


def seeded_objective(objective_fn, base_seed: int, deterministic: bool = True):
    """Wrap a trial objective so it reseeds every RNG before it runs.

    Applied once at the optimizer dispatch point, this covers all backends and
    both execution modes: in-process trials get their RNG state reset so trial
    *n* is not perturbed by trials ``0..n-1``, and worker-process trials get
    seeded at all.

    Args:
        objective_fn: Callable ``(**params) -> float``.
        base_seed: The experiment's ``random_state``.
        deterministic: Forwarded to :func:`set_global_seed`.

    Returns:
        A callable with the same signature.
    """
    import functools

    @functools.wraps(objective_fn)
    def seeded(**params):
        set_global_seed(seed_for_params(base_seed, params), deterministic=deterministic)
        return objective_fn(**params)

    return seeded
