"""Raw Intel RAPL + NVML energy backend -- highest fidelity, Linux + Intel only.

This is the backend the E-LQO paper's methodology corresponds to: direct
hardware counters, no power model anywhere. It exists for two reasons -- as the
cross-check that validates the CodeCarbon backend, and for runs where the
sampling-thread overhead of CodeCarbon is unwanted.

How it differs from CodeCarbon, and why that is sometimes better
----------------------------------------------------------------
* **Counters, not samples.** RAPL ``energy_uj`` and NVML's
  ``nvmlDeviceGetTotalEnergyConsumption`` are monotonically accumulating
  registers. Reading them at the two ends of an interval yields the exact energy
  over that interval, with no sampling thread and no interpolation. A trial
  shorter than a sampling period is therefore measured exactly rather than
  approximated -- the failure mode that forces CodeCarbon's ``quality="low"``
  flag simply does not arise.
* **No CO2 conversion of its own.** Carbon intensity is supplied by the caller
  (``carbon_intensity_g_per_kwh``) rather than looked up, so the number is
  explicit and auditable instead of depending on a bundled table's vintage.

Domain arithmetic -- the trap
------------------------------
``package-N`` **already includes** its ``core`` and ``uncore`` children. Summing
package + core + uncore double-counts almost everything. CPU energy is therefore
the sum over *package* domains only (plural: a dual-socket node has
``intel-rapl:0`` and ``intel-rapl:1``), and ``dram``, where present, is a
sibling rail added separately.

Counter wraparound
------------------
``energy_uj`` wraps at ``max_energy_range_uj`` (262 kJ on an i9-12900K: one
wrap per ~18 min at its 241 W PL2; ~65 kJ on some parts). Two reads can undo at
most one wrap, and a SEARCH phase routinely spans many, so a background sampler
reads every domain at least 4x per worst-case wrap period (``poll_s``, default
10 s, capped at range / 1000 W / 4) and folds wrap-corrected deltas into
monotonic totals; phase and task energies are differences of those totals. A
read gap longer than one worst-case wrap period stamps the reading ``low``.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
import warnings

from pdmlabs.energy.base import BaseEnergyBackend
from pdmlabs.energy.reading import EnergyReading

_log = logging.getLogger(__name__)

_POWERCAP = "/sys/class/powercap"


def _read_int(path):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def discover_rapl():
    """Map the readable RAPL topology.

    Returns ``{"packages": [...], "dram": [...], "readable": bool}`` where each
    entry is ``{"path", "name", "max"}``.
    """
    packages, dram = [], []
    for d in sorted(glob.glob(os.path.join(_POWERCAP, "intel-rapl:*"))):
        base = os.path.basename(d)
        name = None
        try:
            with open(os.path.join(d, "name")) as fh:
                name = fh.read().strip()
        except OSError:
            continue
        entry = {"path": d, "name": name,
                 "max": _read_int(os.path.join(d, "max_energy_range_uj"))}
        # Top-level "intel-rapl:N" is a package; "intel-rapl:N:M" is a subdomain.
        if base.count(":") == 1 and name.startswith("package"):
            packages.append(entry)
        elif name == "dram":
            dram.append(entry)
    readable = any(os.access(os.path.join(p["path"], "energy_uj"), os.R_OK)
                   for p in packages)
    return {"packages": packages, "dram": dram, "readable": readable}


def _delta(prev, cur, maxv):
    """Counter delta in uJ, accounting for wraparound at *maxv*."""
    if prev is None or cur is None:
        return None
    if cur >= prev:
        return cur - prev
    return (maxv - prev + cur) if maxv else None


class _Sample:
    __slots__ = ("t", "pkg", "dram", "gpu", "late")

    def __init__(self, t, pkg, dram, gpu, late=0):
        self.t, self.pkg, self.dram, self.gpu, self.late = t, pkg, dram, gpu, late


class _Accumulator:
    """Wrap-free running totals (uJ) for a set of RAPL counters.

    A pair of reads can undo at most one wrap. Folding every read's
    wrap-corrected delta into a monotonic total -- with reads at least once per
    wrap period, which the background sampler guarantees -- makes any
    interval's energy a plain difference of totals, however often the raw
    counter wrapped in between. Thread-safe: the sampler and the measuring
    thread both call :meth:`update`.
    """

    def __init__(self, domains, guard_s, read=None, clock=None):
        self._domains = domains
        self._read = read or _read_int
        self._clock = clock or time.perf_counter
        self._guard_s = guard_s
        self._lock = threading.Lock()
        n = len(domains)
        self._last = [None] * n
        self._total = [0] * n
        self._broken = [False] * n
        self._t_last = None
        self.n_updates = 0
        self.n_late = 0

    def update(self):
        """Read every counter once; return ``(totals_uj, n_late)``."""
        with self._lock:
            now = self._clock()
            if self._t_last is not None and now - self._t_last > self._guard_s:
                # Longer than one worst-case wrap period since the last read
                # (sampler starved or stopped): a wrap may have been missed.
                self.n_late += 1
            for i, d in enumerate(self._domains):
                if self._broken[i]:
                    continue
                cur = self._read(os.path.join(d["path"], "energy_uj"))
                if cur is None:
                    continue                  # transient; next read covers the gap
                if self._last[i] is not None:
                    step = _delta(self._last[i], cur, d["max"])
                    if step is None:          # wrapped, range unknown: stop trusting it
                        self._broken[i] = True
                        continue
                    self._total[i] += step
                self._last[i] = cur
            self._t_last = now
            self.n_updates += 1
            return ([None if (b or last is None) else tot
                     for tot, last, b in zip(self._total, self._last, self._broken)],
                    self.n_late)


class _Sampler(threading.Thread):
    """Daemon thread that keeps an :class:`_Accumulator` inside one wrap period."""

    def __init__(self, acc, period_s):
        super().__init__(name="pdmlabs-rapl-sampler", daemon=True)
        self._acc, self._period_s = acc, period_s
        self._halt = threading.Event()

    def run(self):
        while not self._halt.wait(self._period_s):
            try:
                self._acc.update()
            except Exception as exc:          # never let sampling die silently
                _log.debug("energy: RAPL sampler read failed: %s", exc)

    def halt(self):
        self._halt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=5.0)


class RaplBackend(BaseEnergyBackend):
    name = "rapl"

    #: Power no single RAPL domain can exceed; far above real parts (the
    #: i9-12900K's PL2 is 241 W) so the derived poll period stays safe.
    POWER_CEILING_W = 1000.0

    def __init__(self, gpu_ids=None, carbon_intensity_g_per_kwh=None,
                 poll_s=10.0, power_ceiling_w=None, **_extra):
        self.topo = discover_rapl()
        self.carbon_intensity_g_per_kwh = carbon_intensity_g_per_kwh
        self._nvml = None
        self._handles = []
        self._gpu_ids = gpu_ids
        self._init_nvml(gpu_ids)
        self._phase0 = None
        self._task0 = None
        domains = self.topo["packages"] + self.topo["dram"]
        maxes = [d["max"] for d in domains if d.get("max")]
        ceiling = float(power_ceiling_w or self.POWER_CEILING_W)
        # Fastest possible wrap: smallest range drained at the ceiling power.
        wrap_s = (min(maxes) / 1e6 / ceiling) if maxes else float("inf")
        self.poll_s = min(float(poll_s), wrap_s / 4.0)   # >= 4 reads per wrap
        self._acc = _Accumulator(domains, guard_s=wrap_s)
        self._sampler = None

    # ------------------------------------------------------------------ #

    def _init_nvml(self, gpu_ids):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            n = pynvml.nvmlDeviceGetCount()
            if gpu_ids is None:
                raw = os.environ.get("CUDA_VISIBLE_DEVICES")
                gpu_ids = ([int(x) for x in raw.split(",") if x.strip().isdigit()]
                           if raw else list(range(n)))
            self._gpu_ids = [i for i in gpu_ids if 0 <= i < n]
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self._gpu_ids]
        except Exception as exc:
            _log.info("energy: NVML unavailable (%s); GPU energy will not be recorded", exc)
            self._nvml, self._handles, self._gpu_ids = None, [], []

    def _gpu_mj(self):
        """Total energy across tracked GPUs, in millijoules, or None."""
        if not self._handles:
            return None
        total = 0
        for h in self._handles:
            try:
                total += self._nvml.nvmlDeviceGetTotalEnergyConsumption(h)
            except Exception:
                return None
        return total

    def _snapshot(self):
        totals, late = self._acc.update()
        n = len(self.topo["packages"])
        return _Sample(time.perf_counter(), totals[:n], totals[n:], self._gpu_mj(), late)

    def _measure(self, start):
        end = self._snapshot()
        dur = end.t - start.t

        # Differences of wrap-free totals: no wrap arithmetic needed here.
        cpu_uj, ok = 0, False
        for a, b in zip(start.pkg, end.pkg):
            if a is not None and b is not None:
                cpu_uj += b - a
                ok = True
        cpu_j = (cpu_uj / 1e6) if ok else None

        dram_j, dok = 0.0, False
        for a, b in zip(start.dram, end.dram):
            if a is not None and b is not None:
                dram_j += (b - a) / 1e6
                dok = True
        ram_j = dram_j if dok else None

        gpu_j = None
        if start.gpu is not None and end.gpu is not None:
            gpu_j = max(0.0, (end.gpu - start.gpu) / 1000.0)

        parts = [x for x in (cpu_j, gpu_j, ram_j) if x is not None]
        total_j = sum(parts) if parts else None
        co2e_g = None
        if total_j is not None and self.carbon_intensity_g_per_kwh:
            co2e_g = total_j / 3.6e6 * float(self.carbon_intensity_g_per_kwh)

        return EnergyReading(
            duration_s=dur, cpu_j=cpu_j, gpu_j=gpu_j, ram_j=ram_j,
            total_j=total_j, co2e_g=co2e_g,
            power_source_cpu="intel_rapl" if ok else "none",
            power_source_gpu="nvml_energy" if gpu_j is not None else "none",
            ram_source="rapl_dram" if dok else "none",
            backend=self.name,
            # "low": a read gap longer than one worst-case wrap period fell
            # inside this interval, so a wrap could have been missed.
            quality=("failed" if not ok else "low" if end.late > start.late else "ok"),
            carbon_intensity_g_per_kwh=self.carbon_intensity_g_per_kwh)

    # ------------------------------------------------------------------ #

    def probe(self):
        return dict(
            available=self.topo["readable"],
            power_source_cpu="intel_rapl" if self.topo["readable"] else "none",
            power_source_gpu="nvml_energy" if self._handles else "none",
            ram_source="rapl_dram" if self.topo["dram"] else "none",
            backend=self.name,
            n_packages=len(self.topo["packages"]),
            n_dram_domains=len(self.topo["dram"]),
            gpu_ids=list(self._gpu_ids or []))

    # The sampler runs only while a phase or task is open, so a backend that is
    # never shut down (one per PdMExperiment) does not leave a thread behind.
    def _ensure_sampler(self):
        if (self._sampler is None or not self._sampler.is_alive()) and self._acc._domains:
            self._sampler = _Sampler(self._acc, self.poll_s)
            self._sampler.start()

    def _release_sampler(self, force=False):
        if self._sampler is not None and (force or (self._phase0 is None and self._task0 is None)):
            self._sampler.halt()
            self._sampler = None

    def start_phase(self, label):
        self._phase0 = self._snapshot()
        self._ensure_sampler()

    def stop_phase(self):
        s, self._phase0 = self._phase0, None
        try:
            if s is None:
                return EnergyReading(duration_s=0.0, backend=self.name, quality="failed")
            return self._measure(s)
        finally:
            self._release_sampler()

    def start_task(self, label):
        self._task0 = self._snapshot()
        self._ensure_sampler()
        return label

    def stop_task(self, token=None):
        s, self._task0 = self._task0, None
        try:
            if s is None:
                return EnergyReading(duration_s=0.0, backend=self.name, quality="failed")
            return self._measure(s)
        finally:
            self._release_sampler()

    def shutdown(self):
        self._release_sampler(force=True)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self._handles = []
