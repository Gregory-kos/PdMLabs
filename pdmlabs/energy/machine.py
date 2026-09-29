"""Machine fingerprint and idle-power baseline.

Two jobs, both about making energy records *comparable*:

**Fingerprint.** Captures everything that changes how much energy a given
computation costs -- CPU model and topology, frequency governor and turbo
state, BLAS thread limits, which GPUs are visible, and crucially the
*measurement provenance* (is RAPL readable, is there a DRAM rail). Records
carrying different fingerprints must never be pooled: a run on a box without a
``dram`` domain reports systematically lower total energy than one with, for
reasons that have nothing to do with the code being measured.

**Idle baseline.** Energy backends report what the *machine* drew, which
includes everything drawing power while doing nothing useful. On a two-GPU
workstation idle is around 84 W (about 22 W CPU package plus about 62 W of idle
GPU), so a ten-hour CPU-only study accrues roughly 3 MJ attributable to nothing
at all. Reporting only total energy would let idle draw dominate the comparison
between optimizers that differ mainly in how long they run.

Both ``total_j`` and idle-subtracted ``dynamic_j`` are always kept. Neither
replaces the other: total is what the wall socket saw, dynamic is what the work
cost.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import time

_log = logging.getLogger(__name__)

_BLAS_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_SLURM_VARS = ("SLURM_JOB_ID", "SLURM_CPUS_ON_NODE", "SLURM_JOB_NODELIST",
               "SLURM_CPU_BIND", "SLURM_JOB_CPUS_PER_NODE", "SLURM_NTASKS")


def _read(path, default=None):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default


def _cpu_model():
    txt = _read("/proc/cpuinfo", "") or ""
    m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
    return m.group(1).strip() if m else platform.processor() or "unknown"


def _pkg_versions():
    out = {}
    for mod in ("pdmlabs", "numpy", "pandas", "sklearn", "scipy", "mlflow",
                "codecarbon", "torch", "smac", "optuna", "hyperopt", "GPyOpt"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def _gpus():
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml
        pynvml.nvmlInit()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            try:
                cap = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            except Exception:
                cap = None
            gpus.append({"index": i, "name": name, "power_limit_w": cap})
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()
        except Exception:
            driver = None
        pynvml.nvmlShutdown()
        return gpus, driver
    except Exception:
        return [], None


def fingerprint():
    """Describe this machine and its measurement capability."""
    from pdmlabs.energy.rapl_backend import discover_rapl
    topo = discover_rapl()
    gpus, driver = _gpus()
    fp = {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "scaling_driver": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
        "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "smt_active": _read("/sys/devices/system/cpu/smt/active"),
        # Measurement provenance -- the part that decides comparability.
        "rapl_readable": topo["readable"],
        "rapl_n_packages": len(topo["packages"]),
        "rapl_has_dram": bool(topo["dram"]),
        "perf_event_paranoid": _read("/proc/sys/kernel/perf_event_paranoid"),
        "gpus": gpus,
        "nvidia_driver": driver,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "blas_env": {v: os.environ.get(v) for v in _BLAS_VARS},
        "slurm": {v: os.environ.get(v) for v in _SLURM_VARS},
        "versions": _pkg_versions(),
    }
    fp["fingerprint_id"] = fingerprint_id(fp)
    return fp


def fingerprint_id(fp):
    """Stable short hash over only the fields that affect comparability.

    Deliberately excludes volatile context (SLURM job id, package versions of
    things unrelated to measurement) so that two runs on the same node in the
    same state share an id.
    """
    keys = ("hostname", "cpu_model", "cpu_count", "governor", "no_turbo",
            "smt_active", "rapl_readable", "rapl_n_packages", "rapl_has_dram",
            "cuda_visible_devices")
    blob = json.dumps({k: fp.get(k) for k in keys}, sort_keys=True, default=str)
    blob += json.dumps(fp.get("blas_env"), sort_keys=True, default=str)
    blob += json.dumps([g.get("name") for g in fp.get("gpus", [])], default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def warn_if_unpinned(fp):
    """Flag configuration that will confound a cross-optimizer comparison."""
    issues = []
    if not fp.get("rapl_readable"):
        issues.append("RAPL is not readable: CPU energy will be a TDP model, "
                      "making energy a near-deterministic function of runtime.")
    if not fp.get("rapl_has_dram"):
        issues.append("No DRAM RAPL domain: RAM energy is a model estimate.")
    if not any(fp.get("blas_env", {}).values()):
        issues.append("No BLAS thread limits set: libraries default to every "
                      "core, so a method's internal parallelism will be "
                      "confounded with the optimizer under test. Pin "
                      "OMP_NUM_THREADS and friends.")
    if fp.get("governor") not in (None, "performance"):
        issues.append("CPU governor is '%s', not 'performance': clock behaviour "
                      "may drift between runs." % fp.get("governor"))
    for msg in issues:
        _log.warning("Energy fingerprint: %s", msg)
    return issues


# --------------------------------------------------------------------------- #
# Idle baseline
# --------------------------------------------------------------------------- #

def measure_idle(backend="codecarbon", duration_s=60.0, settle_s=10.0,
                 backend_kwargs=None):
    """Measure idle power with the same backend that will measure the study.

    *settle_s* is discarded before measuring: GPU clocks and package
    temperature take tens of seconds to reach steady state, and including that
    ramp biases the baseline. Using the study's own backend matters too -- an
    idle figure from one backend subtracted from another's totals would mix two
    different measurement models.
    """
    from pdmlabs.energy import get_energy_backend
    b = get_energy_backend(backend, **(backend_kwargs or {}))
    try:
        if settle_s > 0:
            time.sleep(settle_s)
        with b.phase("IDLE") as cell:
            time.sleep(duration_s)
        r = cell[0] if cell else None
        if r is None or not r.duration_s:
            raise RuntimeError("idle measurement produced no reading")
        def w(j):
            return None if j is None else j / r.duration_s
        return {
            "idle_w": w(r.total_j),
            "idle_cpu_w": w(r.cpu_j),
            "idle_gpu_w": w(r.gpu_j),
            "idle_ram_w": w(r.ram_j),
            "duration_s": r.duration_s,
            "backend": r.backend,
            "power_source_cpu": r.power_source_cpu,
            "ram_source": r.ram_source,
            "measured_at": time.time(),
        }
    finally:
        b.shutdown()


def measure_idle_bracketed(backend="codecarbon", duration_s=60.0, settle_s=10.0,
                           backend_kwargs=None, drift_tolerance=0.10):
    """Measure idle twice and report the drift between them.

    A baseline is only meaningful if the machine's idle state was stable across
    the run. Drift beyond *drift_tolerance* means ambient temperature or a
    background process moved, and every ``dynamic_j`` derived from it is
    suspect -- so the drift is returned rather than averaged away.
    """
    before = measure_idle(backend, duration_s, settle_s, backend_kwargs)
    after = measure_idle(backend, duration_s, settle_s, backend_kwargs)
    a, b_ = before.get("idle_w"), after.get("idle_w")
    drift = abs(b_ - a) / max(a, b_) if (a and b_) else None
    out = {"before": before, "after": after, "drift_frac": drift,
           "idle_w": (a + b_) / 2.0 if (a and b_) else (a or b_),
           "stable": (drift is not None and drift <= drift_tolerance)}
    if drift is not None and drift > drift_tolerance:
        _log.warning("Idle baseline drifted %.1f%% between the two "
                     "measurements (tolerance %.0f%%). Idle-subtracted energy "
                     "from this session is unreliable.",
                     drift * 100, drift_tolerance * 100)
    return out


def save_machine_context(out_dir, backend="codecarbon", idle=True,
                         idle_duration_s=60.0, backend_kwargs=None):
    """Write ``machine.json`` (and ``idle_baseline.json``) for a study."""
    os.makedirs(out_dir, exist_ok=True)
    fp = fingerprint()
    fp["warnings"] = warn_if_unpinned(fp)
    with open(os.path.join(out_dir, "machine.json"), "w") as fh:
        json.dump(fp, fh, indent=2, default=str)
    base = None
    if idle:
        base = measure_idle(backend, idle_duration_s, backend_kwargs=backend_kwargs)
        base["fingerprint_id"] = fp["fingerprint_id"]
        with open(os.path.join(out_dir, "idle_baseline.json"), "w") as fh:
            json.dump(base, fh, indent=2, default=str)
    return fp, base
