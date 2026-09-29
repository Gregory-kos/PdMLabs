"""Append-only cross-process energy spool.

Sibling of :mod:`pdmlabs.optimization.trial_sink`, deliberately **not** an
extension of it: that sink is a best-only high-water mark which keeps one record
per process and overwrites it, whereas an energy study needs every record.

Invariants inherited from ``trial_sink`` (do not break them)
------------------------------------------------------------
* :class:`EnergySink` holds only ``str``. It is cheap to pickle into the trial
  objective closure and owns no file descriptor, socket or thread.
* :data:`_LOCAL_STATE` is a module global of this *importable* module and is
  never pickled. cloudpickle serialises functions and classes of importable
  modules by reference, so ``record()`` running in a worker touches that
  worker's own state. A registry captured in a closure, or defined in
  ``__main__``, would be copied **by value** into every task: several processes
  would then share one file token and clobber each other.

  Corollary, repeated from ``trial_sink``: **never** call
  ``cloudpickle.register_pickle_by_value("pdmlabs")``.
* Keyed by ``(spool_dir, os.getpid())`` so a *forked* child cannot inherit, and
  then append to, its parent's file.
* ``record()`` never raises. Failing to spool degrades the study; it must not
  abort an optimisation run.

Where this DIVERGES from ``trial_sink``, and why each divergence matters
------------------------------------------------------------------------
1. **No ``atexit.register(cleanup)``.** ``TrialSink.create`` registers one
   because its spool is scratch. **This spool is the study data** -- copying
   that line would delete a multi-day run at interpreter exit, with no error.
   Cleanup is explicit and only valid after a successful export.
2. **One append-only ``.jsonl`` per process.** Not one file per record (that
   burns an inode per trial), and not one shared file: ``O_APPEND`` atomicity is
   **not guaranteed on NFS or Lustre**, which is exactly where a SLURM spool
   lives. One writer per file needs no locking and assumes no atomicity.
3. **Torn-tail tolerance instead of a write-then-rename commit protocol.** Each
   record is one ``json.dumps(...) + "\\n"`` in a single ``write()``. A killed
   worker can corrupt only the last line of its own file; the reader skips such
   lines and *counts* them, because silently dropping them would understate
   total energy with no signal.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import socket
import tempfile
import threading
import time
import uuid

_log = logging.getLogger(__name__)

SPOOL_DIR_ENV_VAR = "PDMLABS_ENERGY_SPOOL_DIR"
_RECORD_GLOB = "erec-*.jsonl"
_MAX_TRACKED_SINKS = 64
SCHEMA_VERSION = 1

_LOCAL_STATE: dict = {}
_LOCAL_STATE_LOCK = threading.Lock()


def _json_scalar(value):
    """Coerce to a JSON-safe finite scalar, or ``None``.

    Readings arrive as ``numpy.float64`` (which ``json`` cannot encode) and a
    backend can hand back NaN for a rail it failed to read. ``allow_nan=False``
    at the write site makes an unsanitised NaN a hard error rather than a
    bare ``NaN`` token that pandas and pyarrow disagree about.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return v if math.isfinite(v) else None


class EnergySink:
    """Spools every energy record from every process into one directory."""

    __slots__ = ("spool_dir", "study_id")

    def __init__(self, spool_dir, study_id):
        self.spool_dir = spool_dir
        self.study_id = study_id

    @classmethod
    def create(cls, study_id, base_dir=None):
        base = base_dir or os.environ.get(SPOOL_DIR_ENV_VAR) or None
        if base:
            spool_dir = os.path.abspath(os.path.join(base, str(study_id)))
            os.makedirs(spool_dir, exist_ok=True)
        else:
            spool_dir = os.path.abspath(tempfile.mkdtemp(prefix="pdmlabs_energy_"))
            _log.warning(
                "EnergySink: %s is unset, so the study spool is %s under $TMPDIR. "
                "On a SLURM compute node $TMPDIR is usually wiped by the job "
                "epilogue, which would destroy the study with no error. Set %s "
                "to a durable path.", SPOOL_DIR_ENV_VAR, spool_dir, SPOOL_DIR_ENV_VAR)
        # Deliberately NO atexit.register(self.cleanup) -- see module docstring.
        return cls(spool_dir, study_id)

    # ----------------------------- worker side ----------------------------- #

    def record(self, **fields):
        """Append one record. Safe from any process, including the main one."""
        try:
            key = (self.spool_dir, os.getpid())
            with _LOCAL_STATE_LOCK:
                st = _LOCAL_STATE.get(key)
                if st is None:
                    if len(_LOCAL_STATE) >= _MAX_TRACKED_SINKS:
                        _LOCAL_STATE.pop(next(iter(_LOCAL_STATE)), None)
                    token = "%s-%d-%s" % (
                        socket.gethostname().split(".")[0].replace("-", ""),
                        os.getpid(), uuid.uuid4().hex[:8])
                    st = _LOCAL_STATE[key] = {
                        "token": token, "seq": 0,
                        "path": os.path.join(self.spool_dir, "erec-%s.jsonl" % token)}
                st["seq"] += 1
                seq, token, path = st["seq"], st["token"], st["path"]

            payload = {}
            for k, v in fields.items():
                payload[k] = v if isinstance(v, (dict, list)) else _json_scalar(v)
            payload.update(schema=SCHEMA_VERSION, study_id=self.study_id,
                           record_id="%s-%06d" % (token, seq), seq=seq,
                           pid=os.getpid(), host=socket.gethostname(),
                           wall_time=time.time())
            line = json.dumps(payload, allow_nan=False) + "\n"
            # open/append/close per record: ~20us, negligible against a trial,
            # and it keeps no fd in module state. A cached fd would be duplicated
            # by fork() and parent and child would then share one file offset.
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
            return True
        except Exception as exc:
            _log.warning("EnergySink: failed to spool a record into %s: %s",
                         self.spool_dir, exc)
            return False

    # -------------------------- main-process side -------------------------- #

    def read_all(self):
        """Return ``(records, n_torn_lines)``. Call only after all workers exit."""
        records, torn = [], 0
        for path in sorted(glob.glob(os.path.join(self.spool_dir, _RECORD_GLOB))):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            torn += 1
            except OSError as exc:
                _log.warning("EnergySink: cannot read %s (%s)", path, exc)
        if torn:
            _log.warning(
                "EnergySink: skipped %d torn record line(s) -- a worker was "
                "killed mid-write. Trial energy totals are INCOMPLETE by that "
                "many records; do not treat sums as exhaustive.", torn)
        records.sort(key=lambda r: (r.get("wall_time") or 0, r.get("record_id") or ""))
        return records, torn

    def cleanup(self):
        """Remove the spool. Call ONLY after a successful export."""
        import shutil
        with _LOCAL_STATE_LOCK:
            for k in [k for k in _LOCAL_STATE if k[0] == self.spool_dir]:
                del _LOCAL_STATE[k]
        shutil.rmtree(self.spool_dir, ignore_errors=True)
