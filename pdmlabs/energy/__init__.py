"""PdMLabs energy and carbon measurement layer.

Backend Registry
----------------
``ENERGY_BACKEND_REGISTRY`` maps string identifiers to zero-argument loader
callables.  Adding a new backend requires only inserting a new entry.

Unlike :mod:`pdmlabs.optimization`, which imports every adapter eagerly, the
loaders here are **lazy**: ``codecarbon`` and ``nvidia-ml-py`` are optional
extras, so ``import pdmlabs.energy`` must succeed without them. Importing them
at module scope would make the whole package unimportable on a machine that
only ever runs with tracking disabled.

Supported identifiers
~~~~~~~~~~~~~~~~~~~~~

``"noop"``
    Wall-clock only, zero overhead. **Default**, so existing callers are
    unaffected.

``"codecarbon"``
    CPU + GPU + RAM, kWh and gCO2eq, grid carbon intensity.
    Requires ``pip install pdmlabs[energy]``.

``"rapl"``
    Raw ``/sys/class/powercap`` deltas plus NVML's exact energy counter.
    Highest fidelity, Linux + Intel only.
    Requires ``pip install pdmlabs[energy]``.

Measurement is never assumed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Every backend reports provenance on every reading, and
:meth:`~pdmlabs.energy.reading.EnergyReading.is_measured` is true only when CPU
energy came from ``intel_rapl``. On a kernel that restricts RAPL (the default
since CVE-2020-8694) CodeCarbon falls back to a TDP x utilisation model
*without raising*, which would quietly reduce an energy study to a timing
study. Use :func:`preflight` to gate on that rather than trusting the numbers.
"""

from __future__ import annotations

import logging

from pdmlabs.energy.base import BaseEnergyBackend
from pdmlabs.energy.reading import EnergyReading, KWH_TO_J

_log = logging.getLogger(__name__)

NOOP = "noop"
CODECARBON = "codecarbon"
RAPL = "rapl"


def _load_noop():
    from pdmlabs.energy.noop_backend import NoopEnergyBackend
    return NoopEnergyBackend


def _load_codecarbon():
    from pdmlabs.energy.codecarbon_backend import CodeCarbonBackend
    return CodeCarbonBackend


def _load_rapl():
    from pdmlabs.energy.rapl_backend import RaplBackend
    return RaplBackend


ENERGY_BACKEND_REGISTRY: dict[str, callable] = {
    NOOP:       _load_noop,
    CODECARBON: _load_codecarbon,
    RAPL:       _load_rapl,
}


def get_energy_backend(name, **kwargs) -> BaseEnergyBackend:
    """Instantiate and return the energy backend for *name*.

    Parameters
    ----------
    name:
        One of the keys in ``ENERGY_BACKEND_REGISTRY``. ``None`` is accepted and
        maps to ``"noop"`` so callers can pass an unset option straight through.
    **kwargs:
        Backend-specific configuration (e.g. ``measure_power_secs``,
        ``country_iso_code``, ``gpu_ids``).

    Raises
    ------
    ValueError
        If *name* is not a registered backend identifier.
    ImportError
        If the backend's optional dependency is missing. The message names the
        extra to install.
    """
    if name is None:
        name = NOOP
    if name not in ENERGY_BACKEND_REGISTRY:
        raise ValueError(
            "Unknown energy backend '%s'. Supported identifiers: %s"
            % (name, sorted(ENERGY_BACKEND_REGISTRY.keys()))
        )
    try:
        # Construction is inside the guard too: backends import their optional
        # dependency in __init__ so that a missing extra fails here, with an
        # actionable message, rather than mid-experiment.
        cls = ENERGY_BACKEND_REGISTRY[name]()
        return cls(**kwargs)
    except ImportError as exc:
        raise ImportError(
            "Energy backend '%s' requires optional dependencies that are not "
            "installed. Install them with: pip install pdmlabs[energy]  "
            "(underlying error: %s)" % (name, exc)
        ) from exc


def preflight(name=CODECARBON, allow_estimated=False, **kwargs):
    """Check what *name* can measure here; refuse to proceed on an estimate.

    This is the gate that keeps an energy study from silently becoming a timing
    study. When CPU energy is not from ``intel_rapl`` the result is a model
    whose value is a near-deterministic function of wall-clock time, so
    cross-optimizer conclusions would reduce to "the faster backend won".

    Returns the backend's probe dict. Raises :class:`RuntimeError` when CPU
    energy would be estimated unless *allow_estimated* is explicitly true, in
    which case it warns loudly and stamps the provenance instead.
    """
    backend = get_energy_backend(name, **kwargs)
    try:
        info = backend.probe()
    finally:
        backend.shutdown()
    if name == NOOP:
        return info
    if info.get("power_source_cpu") != "intel_rapl":
        msg = (
            "Energy preflight FAILED: CPU energy would come from '%s', not "
            "'intel_rapl'. Readings would be a TDP x utilisation model, not a "
            "measurement, making energy a near-deterministic function of "
            "runtime. Fix with:\n"
            "  sudo chmod -R a+r /sys/class/powercap/intel-rapl\n"
            "and persist it with a udev rule. Pass allow_estimated=True to "
            "proceed anyway (every record will be stamped as estimated)."
            % info.get("power_source_cpu")
        )
        if not allow_estimated:
            raise RuntimeError(msg)
        _log.warning(msg)
    if info.get("ram_source") != "rapl_dram":
        _log.warning(
            "Energy preflight: RAM energy source is '%s', not 'rapl_dram'. This "
            "CPU exposes no DRAM RAPL domain, so RAM energy is a model estimate. "
            "Do not pool these records with records from a machine that has one.",
            info.get("ram_source"))
    return info


__all__ = ["ENERGY_BACKEND_REGISTRY", "get_energy_backend", "preflight",
           "BaseEnergyBackend", "EnergyReading", "KWH_TO_J",
           "NOOP", "CODECARBON", "RAPL"]
