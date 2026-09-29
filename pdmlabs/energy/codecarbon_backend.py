"""CodeCarbon energy backend -- the default when tracking is enabled.

Gives CPU + GPU + RAM energy, kWh and gCO2eq, and grid carbon intensity.

Design notes, all confirmed empirically against codecarbon 3.3.1
-----------------------------------------------------------------
* **One tracker per phase, tasks inside it.** CodeCarbon's task API is
  single-level, so phases cannot be tasks nested inside a task. Binding a
  tracker's lifetime to a phase gives the property the analysis depends on:
  within a phase, ``sum(task energy) <= phase energy`` *by construction*, which
  is what makes ``E_optimizer_overhead = E_search - sum(E_trial)`` a difference
  of comparable quantities rather than of two unrelated integrals. Phases are
  few (about five per experiment), so per-phase construction costs nothing
  against a search that runs for hours.
* ``stop_task()`` does **not** stop the tracker -- verified -- so many tasks can
  run inside one phase.
* **Unit trap:** ``tracker.stop()`` returns **emissions in kgCO2eq**, not
  energy. Phase energy is ``tracker.final_emissions_data.energy_consumed``
  (kWh). Confusing them silently corrupts every phase total, so this module
  reads the attribute and never the return value.
* **Duration defect (measured, codecarbon 3.3.1).** Once the task API is used,
  ``final_emissions_data.duration`` accumulates only time spent *inside tasks*,
  while ``energy_consumed`` accumulates the whole tracker lifetime. Measured: a
  6.93 s phase containing 1.53 s of tasks reported ``duration == 1.53`` but
  energy covering the full 6.93 s. The energy is right; the duration is not.
  This matters because **idle subtraction multiplies by duration** -- taking
  CodeCarbon's number would have subtracted 1.5 s of idle from a 6.9 s window,
  inflating dynamic energy by roughly the idle draw of five seconds. This
  backend therefore times phases with its own ``perf_counter`` and overrides the
  reported duration. Task durations are unaffected and are left alone.
* ``OfflineEmissionsTracker`` is mandatory on compute nodes: the online tracker
  attempts geolocation and cloud-metadata lookups that block or fail without
  internet.
* ``gpu_ids`` defaults to ``CUDA_VISIBLE_DEVICES``. In machine mode CodeCarbon
  otherwise sums **every** GPU on the node: on a two-GPU box an idle second card
  contributed 69% of all recorded energy during a pure-CPU workload.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
import time

from pdmlabs.energy.base import BaseEnergyBackend
from pdmlabs.energy.reading import EnergyReading

_log = logging.getLogger(__name__)

_CC_LOGGERS = ("codecarbon", "codecarbon.core", "codecarbon.external",
               "codecarbon.output_methods", "apscheduler")


def _silence_codecarbon():
    """Quieten CodeCarbon's per-sample INFO chatter.

    ``pdmlabs.experiment.experiment`` calls ``logging.basicConfig(level=INFO)``
    at import, so at one sample per second a multi-hour search would otherwise
    emit tens of thousands of lines and bury the run's real output.
    """
    for n in _CC_LOGGERS:
        lg = logging.getLogger(n)
        lg.setLevel(logging.ERROR)
        lg.propagate = False


def _default_gpu_ids():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return None
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or None


class CodeCarbonBackend(BaseEnergyBackend):
    name = "codecarbon"

    def __init__(self, country_iso_code="GRC", measure_power_secs=1.0,
                 tracking_mode="machine", gpu_ids=None, rapl_include_dram=True,
                 pue=1.0, output_dir=None, project_name="pdmlabs",
                 min_task_duration_s=None, **_extra):
        # Import eagerly so a missing optional extra fails HERE, at
        # get_energy_backend(), with the actionable "pip install pdmlabs[energy]"
        # message from the registry -- rather than surfacing much later as a raw
        # ImportError from inside the first start_phase(), by which point an
        # experiment is already under way.
        import codecarbon  # noqa: F401

        _silence_codecarbon()
        self.country_iso_code = country_iso_code
        self.measure_power_secs = float(measure_power_secs)
        self.tracking_mode = tracking_mode
        self.gpu_ids = gpu_ids if gpu_ids is not None else _default_gpu_ids()
        self.rapl_include_dram = rapl_include_dram
        self.pue = pue
        self.project_name = project_name
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="pdmlabs_cc_")
        # Below one sampling interval a reading is interpolated rather than
        # sampled; such records stay usable but are stamped quality="low".
        self.min_task_duration_s = (self.measure_power_secs
                                    if min_task_duration_s is None
                                    else float(min_task_duration_s))
        self._tracker = None
        self._phase_label = None
        self._phase_t0 = None
        self._task_label = None
        self._provenance = None
        self._gap_unflushed = False

    # ------------------------------------------------------------------ #

    def _new_tracker(self):
        from codecarbon import OfflineEmissionsTracker
        kwargs = dict(
            country_iso_code=self.country_iso_code,
            measure_power_secs=self.measure_power_secs,
            tracking_mode=self.tracking_mode,
            save_to_file=False, save_to_api=False, save_to_logger=False,
            log_level="error", output_dir=self.output_dir,
            project_name=self.project_name, allow_multiple_runs=True,
            rapl_include_dram=self.rapl_include_dram, pue=self.pue,
        )
        if self.gpu_ids is not None:
            kwargs["gpu_ids"] = self.gpu_ids
        return OfflineEmissionsTracker(**kwargs)

    @staticmethod
    def _extract_provenance(tracker):
        """Read what the tracker actually bound to, for per-record provenance.

        ``CPU._mode`` is the single authoritative signal for measured-vs-modelled.
        It must never be inferred from the numbers: the ``cpu_load`` fallback is
        utilisation-dependent too, which is precisely how it gets mistaken for a
        real reading.
        """
        prov = dict(power_source_cpu="unknown", power_source_gpu="none",
                    ram_source="heuristic", backend="codecarbon")
        try:
            for hw in getattr(tracker, "_hardware", []):
                cn = type(hw).__name__
                if cn == "CPU":
                    prov["power_source_cpu"] = getattr(hw, "_mode", None) or "unknown"
                elif cn == "GPU":
                    prov["power_source_gpu"] = "nvml_power"
                elif cn == "RAM":
                    prov["ram_source"] = "heuristic"
        except Exception as exc:
            _log.warning("energy: could not read CodeCarbon provenance: %s", exc)
        return prov

    def probe(self):
        t = self._new_tracker()
        try:
            t.start()
            prov = self._extract_provenance(t)
        finally:
            try:
                t.stop()
            except Exception:
                pass
        prov["available"] = True
        return prov

    # ------------------------------ phases ----------------------------- #

    def start_phase(self, label):
        if self._tracker is not None:
            _log.warning("energy: start_phase(%s) while phase %s is open; "
                         "closing the previous one.", label, self._phase_label)
            self.stop_phase()
        self._tracker = self._new_tracker()
        self._phase_label = label
        self._tracker.start()
        # Authoritative phase duration: see the duration defect in the module
        # docstring. CodeCarbon's own figure counts only in-task time.
        self._phase_t0 = time.perf_counter()
        self._provenance = self._extract_provenance(self._tracker)

    def stop_phase(self):
        t, label, t0 = self._tracker, self._phase_label, self._phase_t0
        self._tracker, self._phase_label, self._task_label = None, None, None
        self._phase_t0 = None
        if t is None:
            return EnergyReading(duration_s=0.0, backend=self.name, quality="failed")
        wall = None if t0 is None else time.perf_counter() - t0
        unflushed, self._gap_unflushed = self._gap_unflushed, False
        try:
            # Return value is kgCO2eq, NOT energy -- deliberately discarded.
            t.stop()
            d = t.final_emissions_data
            r = self._to_reading(d, self._provenance or {}, duration_override=wall)
            if unflushed and r.quality == "ok":
                r = dataclasses.replace(r, quality="low")
            return r
        except Exception as exc:
            _log.warning("energy: stop_phase(%s) failed: %s", label, exc)
            return EnergyReading(duration_s=0.0, backend=self.name, quality="failed")

    # ------------------------------ tasks ------------------------------ #

    def start_task(self, label):
        if self._tracker is None:
            _log.warning("energy: start_task(%s) with no open phase; the task "
                         "will not be measured. Open a phase envelope first.", label)
            return None
        if self._task_label is not None:
            _log.warning("energy: start_task(%s) while task %s is open; closing "
                         "the previous one so its energy is not misattributed.",
                         label, self._task_label)
            try:
                self._tracker.stop_task()
            except Exception:
                pass
        # CodeCarbon's start_task() pauses the sampler and re-baselines every
        # counter (RAPLFile.start(), GPU last_energy, _last_measured_time)
        # WITHOUT first accumulating what was drawn since the last sample, so
        # the energy of the gap before each task -- the optimizer overhead --
        # would silently vanish from the phase total. Book it first.
        self._flush_pending()
        self._tracker.start_task(label)
        self._task_label = label
        return label

    def _flush_pending(self):
        """Accumulate energy drawn since the tracker's last sample into its totals."""
        t = self._tracker
        measure = getattr(t, "_measure_power_and_energy", None)   # codecarbon >= 2.x, private
        if measure is None:
            measure = getattr(t, "flush", None)                     # public fallback
        if measure is None:
            if not self._gap_unflushed:
                _log.warning("energy: this CodeCarbon version exposes neither "
                             "_measure_power_and_energy nor flush(); energy between "
                             "tasks will be under-counted and phases are stamped "
                             "quality='low'.")
            self._gap_unflushed = True
            return
        try:
            measure()
        except Exception as exc:
            self._gap_unflushed = True
            _log.warning("energy: could not book inter-task energy (%s); phase "
                         "stamped quality='low'.", exc)

    def stop_task(self, token=None):
        if self._tracker is None or self._task_label is None:
            return EnergyReading(duration_s=0.0, backend=self.name, quality="failed")
        self._task_label = None
        try:
            d = self._tracker.stop_task()
            return self._to_reading(d, self._provenance or {})
        except Exception as exc:
            _log.warning("energy: stop_task failed: %s", exc)
            return EnergyReading(duration_s=0.0, backend=self.name, quality="failed")

    # ------------------------------------------------------------------ #

    def _to_reading(self, data, prov, duration_override=None):
        if data is None:
            return EnergyReading(duration_s=0.0, backend=self.name, quality="failed")
        dur = (float(duration_override) if duration_override is not None
               else float(getattr(data, "duration", 0.0) or 0.0))
        total_kwh = getattr(data, "energy_consumed", None)
        co2e_kg = getattr(data, "emissions", None)
        intensity = None
        try:
            if total_kwh and co2e_kg is not None and float(total_kwh) > 0:
                intensity = float(co2e_kg) * 1000.0 / float(total_kwh)
        except (TypeError, ValueError, ZeroDivisionError):
            intensity = None

        quality = "ok"
        if dur > 0 and not total_kwh:
            quality = "zero_energy"
        elif dur < self.min_task_duration_s:
            quality = "low"

        return EnergyReading.from_kwh(
            duration_s=dur,
            cpu_kwh=getattr(data, "cpu_energy", None),
            gpu_kwh=getattr(data, "gpu_energy", None),
            ram_kwh=getattr(data, "ram_energy", None),
            total_kwh=total_kwh, co2e_kg=co2e_kg,
            carbon_intensity_g_per_kwh=intensity, quality=quality, **prov)

    def shutdown(self):
        if self._tracker is not None:
            try:
                self.stop_phase()
            except Exception:
                pass
