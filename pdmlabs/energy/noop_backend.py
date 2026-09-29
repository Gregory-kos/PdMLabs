"""Zero-overhead backend used whenever energy tracking is disabled.

This is the default. It records wall-clock only, so instrumenting the
experiment layer costs nothing measurable when the feature is off, and the
whole plumbing layer (sink, context, decorator, phase envelopes) can be tested
end to end without any measurement hardware or CodeCarbon installed.
"""

from __future__ import annotations

import time

from pdmlabs.energy.base import BaseEnergyBackend
from pdmlabs.energy.reading import EnergyReading

_NONE_PROVENANCE = dict(power_source_cpu="none", power_source_gpu="none",
                        ram_source="none", backend="noop")


class NoopEnergyBackend(BaseEnergyBackend):
    name = "noop"

    def __init__(self, **_kwargs):
        self._phase_t0 = None
        self._task_t0 = None

    def probe(self):
        return dict(available=True, **_NONE_PROVENANCE)

    def start_phase(self, label):
        self._phase_t0 = time.perf_counter()

    def stop_phase(self):
        t0, self._phase_t0 = self._phase_t0, None
        return EnergyReading(duration_s=0.0 if t0 is None else time.perf_counter() - t0,
                             **_NONE_PROVENANCE)

    def start_task(self, label):
        self._task_t0 = time.perf_counter()
        return label

    def stop_task(self, token=None):
        t0, self._task_t0 = self._task_t0, None
        return EnergyReading(duration_s=0.0 if t0 is None else time.perf_counter() - t0,
                             **_NONE_PROVENANCE)
