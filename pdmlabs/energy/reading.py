"""The value type returned by every energy backend.

Canonical unit is the **joule**. kWh is kept as a convenience view but is never
the source of truth: a sub-second trial is O(1e-9) kWh, where float noise and
downstream ``round()`` calls silently destroy precision. Joules keep every
measured quantity in a range where addition is exact enough to sum thousands of
trials without drift.
"""

from __future__ import annotations

import dataclasses
import math

KWH_TO_J = 3.6e6


def _as_j(kwh):
    """kWh -> J, mapping None/NaN/inf to None rather than propagating them."""
    if kwh is None:
        return None
    try:
        v = float(kwh)
    except (TypeError, ValueError):
        return None
    return v * KWH_TO_J if math.isfinite(v) else None


@dataclasses.dataclass(frozen=True)
class EnergyReading:
    """One energy measurement over a closed time interval.

    Every field that can be absent is ``None`` rather than ``0.0``. The
    distinction matters: a missing DRAM rail and a rail that genuinely drew no
    energy must not be summed into the same total, or a machine without a
    ``dram`` RAPL domain would silently report lower total energy than one with.
    """

    duration_s: float
    cpu_j: float | None = None
    gpu_j: float | None = None
    ram_j: float | None = None
    total_j: float | None = None
    co2e_g: float | None = None

    # --- provenance: the fields that separate a measurement from an estimate ---
    power_source_cpu: str = "unknown"   # "intel_rapl" | "cpu_load" | "constant" | "none"
    power_source_gpu: str = "unknown"   # "nvml_energy" | "nvml_power" | "none"
    ram_source: str = "unknown"         # "rapl_dram" | "heuristic" | "none"
    backend: str = "unknown"
    quality: str = "ok"                 # "ok" | "low" | "zero_energy" | "failed"
    carbon_intensity_g_per_kwh: float | None = None

    @property
    def total_kwh(self):
        return None if self.total_j is None else self.total_j / KWH_TO_J

    @property
    def avg_power_w(self):
        if self.total_j is None or not self.duration_s:
            return None
        return self.total_j / self.duration_s

    @property
    def is_measured(self):
        """True only when CPU energy came from hardware counters.

        This is the single predicate that decides whether a run is a measurement
        or a model. Nothing may infer it from the shape of the numbers -- a
        TDP*utilisation model also varies with load, which is exactly how such a
        fallback gets mistaken for a real reading.
        """
        return self.power_source_cpu == "intel_rapl"

    @classmethod
    def from_kwh(cls, duration_s, cpu_kwh=None, gpu_kwh=None, ram_kwh=None,
                 total_kwh=None, co2e_kg=None, **provenance):
        """Build from CodeCarbon's kWh/kg convention."""
        cpu_j, gpu_j, ram_j = _as_j(cpu_kwh), _as_j(gpu_kwh), _as_j(ram_kwh)
        total_j = _as_j(total_kwh)
        if total_j is None:
            parts = [p for p in (cpu_j, gpu_j, ram_j) if p is not None]
            total_j = sum(parts) if parts else None
        co2e_g = None
        if co2e_kg is not None:
            try:
                v = float(co2e_kg)
                co2e_g = v * 1000.0 if math.isfinite(v) else None
            except (TypeError, ValueError):
                co2e_g = None
        return cls(duration_s=float(duration_s), cpu_j=cpu_j, gpu_j=gpu_j,
                   ram_j=ram_j, total_j=total_j, co2e_g=co2e_g, **provenance)

    def dynamic_j(self, idle_w, clamp=True):
        """Idle-subtracted energy, and whether the clamp fired.

        Returns ``(value, clamped)``. Short or noisy intervals can yield a
        negative result; clamping to zero is right, but doing so *silently* is
        not -- the clamp rate is itself a data-quality signal, so it is returned
        rather than swallowed.
        """
        if self.total_j is None or idle_w is None:
            return None, False
        v = self.total_j - float(idle_w) * self.duration_s
        if v < 0 and clamp:
            return 0.0, True
        return v, False

    def as_dict(self):
        d = dataclasses.asdict(self)
        d["total_kwh"] = self.total_kwh
        d["avg_power_w"] = self.avg_power_w
        d["is_measured"] = self.is_measured
        return d
