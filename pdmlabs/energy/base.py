"""Abstract base class for all PdMLabs energy backends.

Mirrors :class:`pdmlabs.optimization.base.BaseOptimizerAdapter`: backends
translate a vendor API (CodeCarbon, raw RAPL/NVML) into one uniform contract so
the experiment layer never imports a vendor package directly.

Nesting contract
----------------
A backend supports exactly **one open phase** and, inside it, **one open task**.
CodeCarbon's task API is single-level, so this interface deliberately does not
promise arbitrary nesting -- promising it would make the CodeCarbon backend a
lie rather than an implementation.
"""

from __future__ import annotations

import abc
import contextlib
import logging
import time

from pdmlabs.energy.reading import EnergyReading

_log = logging.getLogger(__name__)


class BaseEnergyBackend(abc.ABC):
    """Uniform start/stop energy measurement.

    Implementations must honour three rules, because this object runs inside an
    HPO trial and must never be able to break one:

    1. ``stop_*`` returns an :class:`EnergyReading`, never raises. A measurement
       failure degrades the study; it must not abort an optimisation run.
    2. Provenance is always populated, even on failure, so no record can be
       mistaken for a hardware measurement.
    3. ``start_task`` while a task is open is an error the backend resolves
       itself (close-and-warn), never a silent overwrite that would attribute
       one task's energy to another.
    """

    name: str = "base"

    @abc.abstractmethod
    def probe(self) -> dict:
        """Report what this backend can actually measure, without measuring.

        Returns a dict with at least ``power_source_cpu``, ``power_source_gpu``,
        ``ram_source`` and ``available`` (bool). Called by the preflight gate.
        """

    @abc.abstractmethod
    def start_phase(self, label: str) -> None: ...

    @abc.abstractmethod
    def stop_phase(self) -> EnergyReading: ...

    @abc.abstractmethod
    def start_task(self, label: str) -> object | None: ...

    @abc.abstractmethod
    def stop_task(self, token: object | None) -> EnergyReading: ...

    def shutdown(self) -> None:
        """Release any long-lived resource. Idempotent."""

    @contextlib.contextmanager
    def phase(self, label):
        """Measure a phase, yielding a one-element list that receives the reading.

        The reading is delivered through a mutable cell rather than as the
        ``as`` value because the value is only known at ``__exit__``.
        """
        cell = []
        t0 = time.perf_counter()
        try:
            self.start_phase(label)
        except Exception as exc:
            _log.warning("energy: start_phase(%s) failed: %s", label, exc)
        try:
            yield cell
        finally:
            try:
                cell.append(self.stop_phase())
            except Exception as exc:
                _log.warning("energy: stop_phase(%s) failed: %s", label, exc)
                cell.append(EnergyReading(duration_s=time.perf_counter() - t0,
                                          backend=self.name, quality="failed"))
