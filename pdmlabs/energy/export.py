"""Turn the JSONL spool into a tidy, portable, self-describing table.

The spool is an append-only write format optimised for never losing a record
under concurrency. This module turns it into the long-format table the analysis
works from: one row per (study, run, phase-or-trial), with the machine
fingerprint and idle baseline folded in so the file stands alone -- no MLflow,
no spool, no environment needed to read it later.

Two things happen here that cannot happen at write time:

* **Idle subtraction.** ``dynamic_j = total_j - idle_w * duration_s``. It needs
  the idle baseline, which is measured once per study rather than per record.
  Negative results are clamped to zero *and* flagged, because the clamp rate is
  itself a data-quality signal -- silently clamping would hide a bad baseline.
* **The fingerprint gate.** Records from machines that differ in ways that
  affect energy (DRAM rail present or not, RAPL readable or not, different GPUs)
  are not comparable. Pooling them is refused rather than warned about, because
  the resulting number looks perfectly reasonable and is wrong.
"""

from __future__ import annotations

import json
import logging
import os

_log = logging.getLogger(__name__)

# Columns that carry meaning even when null; kept in a stable order so the
# exported file has a predictable schema across studies.
_CORE = ["study_id", "record_id", "kind", "phase", "flavor", "experiment_name",
         "optimizer", "n_jobs", "num_iteration", "optimization_param", "maximize",
         "use_cache", "energy_trial_id", "mlflow_run_id", "mlflow_experiment_id",
         "params_hash", "objective_value", "exit_path", "cached", "error",
         "duration_s", "measured_duration_s", "cpu_j", "gpu_j", "ram_j",
         "total_j", "dynamic_j", "dynamic_clamped", "co2e_g",
         "carbon_intensity_g_per_kwh", "avg_power_w", "power_source_cpu",
         "power_source_gpu", "ram_source", "backend", "quality", "is_measured",
         "attribution", "idle_w", "fingerprint_id", "host", "pid", "seq",
         "wall_time", "t_start_unix"]


def _load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def build_rows(records, idle=None, fingerprint=None):
    """Enrich raw spool records into export rows."""
    idle_w = (idle or {}).get("idle_w")
    fid = (fingerprint or {}).get("fingerprint_id")
    rows = []
    for r in records:
        row = dict(r)
        row.setdefault("fingerprint_id", fid)
        row["idle_w"] = idle_w

        total, dur = row.get("total_j"), row.get("duration_s")
        row["avg_power_w"] = (total / dur) if (total and dur) else None

        dyn, clamped = None, False
        if total is not None and idle_w is not None and dur:
            dyn = total - idle_w * dur
            if dyn < 0:
                dyn, clamped = 0.0, True
        row["dynamic_j"] = dyn
        row["dynamic_clamped"] = clamped
        # How much of this record's energy was idle draw. When this is high the
        # dynamic figure is a small difference of two large numbers, so a few
        # percent of error in the idle baseline becomes tens of percent of
        # error in dynamic_j. Carried per record so the analysis can say which
        # basis is trustworthy instead of silently preferring one.
        row["idle_share"] = ((idle_w * dur / total)
                             if (total and idle_w is not None and dur and total > 0)
                             else None)
        rows.append(row)
    return rows


def check_fingerprints(rows, allow_mixed=False):
    """Refuse to pool records from machines that are not comparable."""
    ids = {r.get("fingerprint_id") for r in rows if r.get("fingerprint_id")}
    if len(ids) > 1 and not allow_mixed:
        raise ValueError(
            "Refusing to export %d different machine fingerprints into one "
            "table: %s. Energy from machines that differ in measurement "
            "capability (DRAM rail, RAPL access, GPU set) is not comparable, "
            "and the pooled numbers would look entirely plausible while being "
            "wrong. Export per fingerprint, or pass allow_mixed=True if you "
            "have a specific reason." % (len(ids), sorted(ids)))
    return ids


def reconcile(rows):
    """Cross-check the invariants the analysis depends on.

    Returns a list of findings, one per SEARCH phase. A list of flat records
    rather than a dict keyed by tuple: it is JSON-serialisable, and it drops
    straight into a DataFrame alongside the exported rows.

    These are reported, not enforced: a violation means the *measurement* is
    untrustworthy, which the analyst must see rather than have silently
    corrected.
    """
    out = []
    trials = [r for r in rows if r.get("kind") == "trial"]
    phases = [r for r in rows if r.get("kind") == "phase"]

    def _key(r):
        # A phase_id (stamped on the phase and on each of its trials) is exact.
        # The dims fallback must separate repeats *and* seeds: without them one
        # repeat's SEARCH phase is matched against every repeat's trials.
        if r.get("phase_id"):
            return (r.get("study_id"), r.get("phase_id"))
        return (r.get("study_id"), r.get("experiment_name"), r.get("optimizer"),
                r.get("repeat_idx"), r.get("seed"), r.get("random_state"))

    for p in phases:
        if p.get("phase") != "SEARCH":
            continue
        key = _key(p)
        mine = [t for t in trials if _key(t) == key and t.get("phase") == "SEARCH"]
        sum_j = sum(t.get("total_j") or 0.0 for t in mine)
        phase_j = p.get("total_j") or 0.0
        # Concurrent (phase_only) trials carry no energy: the residual is not
        # an overhead, so report it as unknown rather than as 100%.
        n_po = sum(1 for t in mine if t.get("attribution") == "phase_only")
        e_opt = None if (n_po or not mine) else phase_j - sum_j
        finding = {
            "study_id": p.get("study_id"), "experiment_name": p.get("experiment_name"),
            "optimizer": p.get("optimizer"), "repeat_idx": p.get("repeat_idx"),
            "seed": p.get("seed"), "random_state": p.get("random_state"),
            "phase_id": p.get("phase_id"),
            "n_trials": len(mine), "n_phase_only": n_po,
            "n_cached": sum(1 for t in mine if t.get("cached")),
            "n_errors": sum(1 for t in mine if t.get("exit_path") == "error"),
            "search_j": phase_j,
            "sum_trial_j": sum_j,
            "e_optimizer_j": e_opt,
            "e_optimizer_frac": (e_opt / phase_j) if (phase_j and e_opt is not None) else None,
            "invariant_ok": None if e_opt is None else e_opt >= -1e-6,
            "all_measured": all(t.get("is_measured") for t in mine) if mine else None,
            "n_low_quality": sum(1 for t in mine if t.get("quality") == "low"),
            "n_unjoined": sum(1 for t in mine if not t.get("mlflow_run_id")),
        }
        out.append(finding)
        if finding["invariant_ok"] is False:
            _log.warning(
                "Reconcile %s/%s (repeat %s): sum(trial energy) EXCEEDS phase "
                "energy by %.2f J. The phase envelope did not enclose every "
                "trial, or they came from different backends. "
                "E_optimizer_overhead is not interpretable for this run.",
                p.get("experiment_name"), p.get("optimizer"), p.get("repeat_idx"), -e_opt)
    return out


def export_study(spool_dir, out_dir=None, study_id=None, allow_mixed=False,
                 cleanup=False):
    """Read a study's spool and write ``records.parquet`` plus ``records.csv``.

    Returns ``(rows, summary)``. The spool is left in place unless *cleanup* is
    explicitly requested -- it is the study's raw data, and deleting it on the
    strength of an export that might itself be wrong is not a trade worth making.
    """
    from pdmlabs.energy.sink import EnergySink

    spool_dir = os.path.abspath(spool_dir)
    study_id = study_id or os.path.basename(spool_dir.rstrip("/"))
    out_dir = out_dir or spool_dir

    sink = EnergySink(spool_dir, study_id)
    records, torn = sink.read_all()
    if not records:
        raise ValueError("No energy records found in %s" % spool_dir)

    parent = os.path.dirname(spool_dir)
    idle = (_load_json(os.path.join(spool_dir, "idle_baseline.json"))
            or _load_json(os.path.join(parent, "idle_baseline.json")))
    fp = (_load_json(os.path.join(spool_dir, "machine.json"))
          or _load_json(os.path.join(parent, "machine.json")))

    rows = build_rows(records, idle, fp)
    check_fingerprints(rows, allow_mixed=allow_mixed)
    summary = {
        "study_id": study_id, "n_records": len(rows), "n_torn": torn,
        "n_trials": sum(1 for r in rows if r.get("kind") == "trial"),
        "n_phases": sum(1 for r in rows if r.get("kind") == "phase"),
        "idle_w": (idle or {}).get("idle_w"),
        "fingerprint_id": (fp or {}).get("fingerprint_id"),
        "has_idle_baseline": idle is not None,
        "n_dynamic_clamped": sum(1 for r in rows if r.get("dynamic_clamped")),
        "reconcile": reconcile(rows),
    }
    shares = [r["idle_share"] for r in rows
              if r.get("kind") == "phase" and r.get("idle_share") is not None]
    if shares:
        worst = max(shares)
        summary["max_idle_share"] = worst
        summary["dynamic_reliable"] = worst < 0.5
        if worst >= 0.5:
            _log.warning(
                "Idle draw is %.0f%% of the energy in at least one phase. "
                "dynamic_j is then a small residual of two large numbers, so a "
                "few percent of drift in the idle baseline becomes tens of "
                "percent in dynamic_j -- measured here as 0.6-2.4%% run-to-run "
                "variation on total energy against 82-109%% on dynamic. Prefer "
                "total_j for cross-optimizer comparison at this idle share, or "
                "cut idle first (exclude unused GPUs via gpu_ids, which are "
                "typically most of it).", 100 * worst)
    if idle is None:
        _log.warning("No idle_baseline.json for study '%s': dynamic_j is null "
                     "and only total energy is available. On a machine with "
                     "idle GPUs that can be most of the recorded energy.", study_id)

    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    cols = _CORE + sorted({k for r in rows for k in r} - set(_CORE))
    df = pd.DataFrame(rows).reindex(columns=cols)
    # params is a dict; keep it as JSON so the schema stays flat and portable.
    if "params" in df.columns:
        df["params"] = df["params"].map(lambda v: json.dumps(v, default=str)
                                        if isinstance(v, dict) else v)
    csv_path = os.path.join(out_dir, "records.csv")
    df.to_csv(csv_path, index=False)
    written = [csv_path]
    try:
        pq = os.path.join(out_dir, "records.parquet")
        df.to_parquet(pq, index=False)
        written.append(pq)
    except Exception as exc:
        _log.warning("Parquet export unavailable (%s); CSV written instead", exc)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    summary["files"] = written

    if cleanup:
        sink.cleanup()
    return rows, summary
