"""Cross-process best-trial spool for PdMLabs experiments.

Why this exists
---------------
Every optimizer backend that honours ``n_jobs > 1`` evaluates the experiment's
``optimization_objective`` in a **separate OS process**:

* ``mango``    -- ``joblib.Parallel(n_jobs)``                 (loky)
* ``gpyopt``   -- ``joblib.Parallel(n_jobs=n_jobs)``          (loky)
* ``optuna``   -- ``joblib.Parallel(backend="loky")``         (loky)
* ``smac``     -- ``Scenario(n_workers=n_jobs)`` -> dask ``Client(processes=True,
                  threads_per_worker=1)``
* ``hyperopt`` -- sequential (main process)

Assignments the objective makes to ``self`` therefore land on a deserialized copy
of the experiment that is discarded when the task ends.  This module is a real IPC
channel: workers spool their best trial into a shared directory and the main
process reads the global best back after the optimizer returns.  Nothing is ever
re-evaluated to recover it.

Design invariants
-----------------
1. :class:`TrialSink` holds only ``str`` / ``bool``.  It is cheap to pickle into
   the objective closure and owns no OS handles, sockets or threads.

2. The per-process high-water mark (:data:`_LOCAL_BEST`) is a **module global of
   this importable module** and is never pickled.  cloudpickle serialises classes
   from importable modules *by reference*, so ``TrialSink.record`` running in a
   worker touches that worker's own registry.  A registry defined in ``__main__``
   or captured in a closure would instead be copied *by value* into every task:
   the high-water mark would never advance and -- worse -- several processes would
   share one file token and clobber each other.
   Corollary: never call ``cloudpickle.register_pickle_by_value("pdmlabs")``.

3. The registry is keyed by ``(spool_dir, os.getpid())`` so a *forked* child
   cannot inherit, and then overwrite, its parent's spool file.

4. Commit protocol: the pipeline blob is written to a versioned name that no
   process has written before, then the metadata file (which *names* that blob) is
   atomically replaced.  A worker killed between the two steps leaves a metadata
   file pointing at the previous, complete blob -- never a torn pair.

5. Two marks per process, in two file families.  ``meta-*.json`` tracks the best
   score seen and is scalars-only; ``pmeta-*.json`` plus its blob tracks the best
   trial that actually carried a pipeline.  A trial can have a score and a
   threshold but no pipeline -- that is what a run served from the MLflow cache
   looks like -- and separating the two means such a trial can win on score
   without costing us the best pipeline we hold.  Because only the pipe family
   owns a blob, this costs no extra pipeline serialisations.

Correctness of the per-process high-water mark
----------------------------------------------
The globally best trial ran in exactly one process, and within that process it was
necessarily that process's best too.  Hence the globally best record is always
among the spooled files.  Non-finite scores are rejected outright: a NaN
high-water mark would make every later comparison ``False`` and silently stop that
process from spooling again.

What a recovered record guarantees
----------------------------------
``score``/``th``/``th_to_rul``/``params`` always describe one real trial: the best
one recorded.  ``pipeline`` is normally that same trial's, and
``pipeline_is_winner`` says so.  When the winner carried no pipeline it falls back
to the best trial that did, and ``pipeline_score``/``pipeline_params``/
``pipeline_th`` describe *that* trial -- so a fallback model is stamped with its
own threshold rather than the winner's.

Ties
----
Several parameterizations can reach the same best objective.  The optimizer picks
one of them as ``best_params`` while the sink independently picks one tied record,
and they need not be the same configuration.  :meth:`TrialSink.best` therefore
accepts ``prefer_params`` and, among tied records, prefers the one whose params
match; failing that it takes the earliest tied trial (deterministic given the file
set) and reports ``matched_best_params=False`` so the caller can say so out loud.
No scheme can guarantee a match: a strict high-water mark keeps the *first* trial
to reach a score, so if the optimizer's pick was a later tie in the same process
that record was never spooled.
"""

from __future__ import annotations

import atexit
import glob
import json
import logging
import math
import os
import pickle
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass

try:                                          # preferred: also handles lambdas
    import cloudpickle as _pickler            # and locally-defined classes
except ImportError:                           # pragma: no cover
    try:
        from joblib.externals import cloudpickle as _pickler
    except ImportError:                       # pragma: no cover
        _pickler = pickle

_log = logging.getLogger(__name__)

SPOOL_DIR_ENV_VAR = "PDMLABS_TRIAL_SPOOL_DIR"
_META_GLOB = "meta-*.json"      # best score per process, scalars only
_PMETA_GLOB = "pmeta-*.json"    # best pipeline-bearing trial per process
_MAX_TRACKED_SINKS = 64  # bounds the registry in long-lived reused workers

# (spool_dir, pid) -> {"token": str, "score": float|None, "seq": int, "blob": str|None}
_LOCAL_BEST = {}
_LOCAL_BEST_LOCK = threading.Lock()


def _as_float(value):
    """Coerce to a plain JSON-safe finite float, or ``None``.

    Metric values arrive as ``numpy.float64``, which ``json`` cannot encode.
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def normalise_params(params):
    """Canonicalise a trial's params for storage and display.

    Values are kept as strings, matching the idiom ``_check_cached_run`` already
    uses against MLflow (``current_run.loc['params.' + k] != str(v)``). Exact
    equality on the raw values is not usable: an adapter may hand back a value
    with a different type than it received.
    """
    if not params:
        return None
    try:
        return {str(k): str(v) for k, v in params.items()}
    except Exception:  # pragma: no cover - params is always a plain dict today
        return None


def _comparable_value(value):
    """Round numeric values so formatting noise does not look like a difference.

    SMAC round-trips ordinals through ``str()``/``float()`` and GPyOpt snaps
    discrete floats, so the same configuration can come back as ``0.3`` or
    ``0.30000000000000004``. Comparing those raw would report a configuration
    mismatch that is not real. Booleans are left alone (``bool`` is an ``int``,
    and ``True``/``1.0`` should stay distinct).
    """
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return repr(round(number, 10)) if math.isfinite(number) else str(value)


def _comparable_params(params):
    """Apply :func:`_comparable_value` across an already-normalised param dict."""
    if not params:
        return None
    return {k: _comparable_value(v) for k, v in params.items()}


@dataclass(frozen=True)
class TrialRecord:
    """The globally best trial recovered from a spool directory.

    ``score``/``th``/``th_to_rul``/``params`` always describe one real trial --
    the best one recorded. ``pipeline`` usually belongs to that same trial, but
    when the winner carried no pipeline (it came from the run cache, or failed to
    serialize) it falls back to the best trial that did produce one;
    ``pipeline_is_winner`` says which, and ``pipeline_score``/``pipeline_params``
    identify the fallback.
    """

    score: float
    th: float = None
    th_to_rul: float = None
    pipeline: object = None
    pipeline_error: str = None
    params: dict = None
    matched_best_params: bool = True
    tied_count: int = 1
    pipeline_is_winner: bool = True
    pipeline_score: float = None
    pipeline_params: dict = None
    pipeline_th: float = None


class TrialSink:
    """Spools each worker process's best trial into a shared directory.

    Instances are created in the main process by :meth:`create`, captured by the
    ``optimization_objective`` closure, and shipped to every worker.  Workers call
    :meth:`record`; the main process calls :meth:`best` once, after
    ``_run_optimizer`` returns, then :meth:`cleanup`.
    """

    __slots__ = ("spool_dir", "maximize")

    def __init__(self, spool_dir, maximize=True):
        self.spool_dir = spool_dir
        self.maximize = bool(maximize)

    # ------------------------------------------------------------------ #
    # Construction / teardown                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def create(cls, maximize=True):
        """Create a fresh, private spool directory.

        ``mkdtemp`` uses ``O_CREAT | O_EXCL``, so the directory is unique against
        every process on the machine -- concurrent experiments cannot collide.
        Set ``PDMLABS_TRIAL_SPOOL_DIR`` to place the spool somewhere other than
        ``$TMPDIR`` (useful when pipelines are large or ``/tmp`` is small).
        """
        base = os.environ.get(SPOOL_DIR_ENV_VAR) or None
        if base:
            try:
                os.makedirs(base, exist_ok=True)
            except OSError as exc:
                _log.warning(
                    "TrialSink: %s=%r is not usable (%s); falling back to $TMPDIR",
                    SPOOL_DIR_ENV_VAR, base, exc,
                )
                base = None
        spool_dir = os.path.abspath(
            tempfile.mkdtemp(prefix="pdmlabs_trials_", dir=base)
        )
        sink = cls(spool_dir, maximize)
        atexit.register(sink.cleanup)
        return sink

    def cleanup(self):
        """Remove the spool directory. Idempotent; safe to call from ``finally``."""
        with _LOCAL_BEST_LOCK:
            for key in [k for k in _LOCAL_BEST if k[0] == self.spool_dir]:
                del _LOCAL_BEST[key]
        shutil.rmtree(self.spool_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    # ------------------------------------------------------------------ #
    # Worker side                                                         #
    # ------------------------------------------------------------------ #

    def record(self, score, pipeline=None, th=None, th_to_rul=None, params=None):
        """Spool this trial if *score* beats this process's high-water marks.

        Two marks are kept per process. The *any* mark tracks the best score
        regardless of whether a pipeline came with it, and is written as a small
        scalars-only ``meta-*.json``; it is authoritative for ``score``/``th``.
        The *pipe* mark tracks the best trial that actually carried a pipeline
        and owns the expensive blob. Keeping them apart means a trial with no
        pipeline (served from the run cache) can still win on score without
        discarding the best pipeline we hold, and costs no extra blob writes.

        Safe to call from any process, including the main one. Returns ``True``
        if anything was committed. Never raises: failing to spool degrades the
        reported result but must not abort an optimization run.
        """
        try:
            score = float(score)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(score):
            # A NaN high-water mark makes every later comparison False, which
            # would silently stop this process from ever spooling again.
            return False

        key = (self.spool_dir, os.getpid())
        with _LOCAL_BEST_LOCK:
            state = _LOCAL_BEST.get(key)
            if state is None:
                if len(_LOCAL_BEST) >= _MAX_TRACKED_SINKS:
                    _LOCAL_BEST.pop(next(iter(_LOCAL_BEST)), None)
                state = _LOCAL_BEST[key] = {
                    "token": "%d-%s" % (os.getpid(), uuid.uuid4().hex),
                    "any_score": None,
                    "pipe_score": None,
                    "seq": 0,
                    "blob": None,
                }
            improved_any = (state["any_score"] is None
                            or self._is_better(score, state["any_score"]))
            improved_pipe = pipeline is not None and (
                state["pipe_score"] is None
                or self._is_better(score, state["pipe_score"]))
            if not improved_any and not improved_pipe:
                return False

            token = state["token"]
            previous_any = state["any_score"]
            previous_blob = state["blob"]
            if improved_any:
                state["any_score"] = score
            if improved_pipe:
                state["seq"] += 1
            seq = state["seq"]

        common = {
            "score": score,
            "th": _as_float(th),
            "th_to_rul": _as_float(th_to_rul),
            "params": normalise_params(params),
            "pid": os.getpid(),
            "wall_time": time.time(),
        }

        blob_name = None
        pipeline_error = None
        if improved_pipe:
            candidate = "blob-%s-%06d.pkl" % (token, seq)
            try:
                self._atomic_write(
                    candidate,
                    lambda fh: _pickler.dump(
                        pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL
                    ),
                )
                blob_name = candidate
            except Exception as exc:
                # Leave the pipe mark untouched so the last pipeline we did
                # manage to store stays available as a fallback.
                pipeline_error = "%s: %s" % (type(exc).__name__, exc)
                _log.warning(
                    "TrialSink: pipeline for trial (score=%r) is not serialisable: %s",
                    score, pipeline_error,
                )

        try:
            if blob_name is not None:
                payload = dict(common, blob=blob_name, seq=seq)
                # Committed after the blob, which it names, so the pair is never
                # torn: a crash leaves a pmeta pointing at a complete blob.
                self._atomic_write("pmeta-%s.json" % token,
                                   lambda fh: fh.write(
                                       json.dumps(payload).encode("utf-8")))
            if improved_any:
                payload = dict(common, pipeline_error=pipeline_error)
                self._atomic_write("meta-%s.json" % token,
                                   lambda fh: fh.write(
                                       json.dumps(payload).encode("utf-8")))
        except Exception as exc:
            with _LOCAL_BEST_LOCK:
                st = _LOCAL_BEST.get(key)
                if st is not None and improved_any:
                    st["any_score"] = previous_any  # let a later trial retry
            _log.warning(
                "TrialSink: failed to spool trial (score=%r) into %s: %s",
                score, self.spool_dir, exc,
            )
            return False

        if blob_name is not None:
            with _LOCAL_BEST_LOCK:
                st = _LOCAL_BEST.get(key)
                if st is not None and st["seq"] == seq:
                    st["pipe_score"] = score
                    st["blob"] = blob_name
            if previous_blob and previous_blob != blob_name:
                try:
                    os.remove(os.path.join(self.spool_dir, previous_blob))
                except OSError:
                    pass
        return True

    # ------------------------------------------------------------------ #
    # Main-process side                                                   #
    # ------------------------------------------------------------------ #

    def best(self, prefer_params=None):
        """Return the globally best spooled trial, or ``None`` if there is none.

        Call only after the optimizer has returned (all workers finished).  Reads
        every small metadata file but unpickles at most one blob.

        *prefer_params* is the optimizer's ``best_params``.  When several trials
        tie on the best score, the record whose params match it is preferred, so
        ``th``/``best_pipeline`` describe the same configuration the optimizer
        reported.  See the module docstring's *Ties* section.
        """
        candidates = self._read_family(_META_GLOB)
        if not candidates:
            # Nothing scored, but a pipeline may still have been stored if the
            # only writes that survived were pipe-family ones.
            candidates = self._read_family(_PMETA_GLOB)
        if not candidates:
            return None

        best_score = (max if self.maximize else min)(c[0] for c in candidates)
        tied = [c for c in candidates if c[0] == best_score]
        # Deterministic given the file set: earliest trial first, then filename.
        tied.sort(key=lambda c: (c[2].get("wall_time") or 0.0, c[1]))

        wanted = _comparable_params(normalise_params(prefer_params))
        matched = True
        chosen = tied[0]
        if wanted is not None:
            for candidate in tied:
                if _comparable_params(candidate[2].get("params")) == wanted:
                    chosen = candidate
                    break
            else:
                # No spooled record carries the optimizer's configuration. This
                # is not limited to len(tied) > 1: a lone record can still be a
                # different config than ``best_params`` (the winning trial came
                # from the run cache, or never spooled). Only claim a mismatch
                # when the chosen record's params are actually known.
                matched = chosen[2].get("params") is None
        score, _, meta = chosen

        # The pipeline lives in its own family, so a pipeline-less winner (one
        # served from the run cache) does not cost us the best pipeline we hold.
        pipeline = None
        pipeline_is_winner = True
        pipeline_score = None
        pipeline_params = None
        pipeline_th = None
        pipe_candidates = self._read_family(_PMETA_GLOB)
        if pipe_candidates:
            pipe_best = (max if self.maximize else min)(c[0] for c in pipe_candidates)
            pipe_tied = [c for c in pipe_candidates if c[0] == pipe_best]
            pipe_tied.sort(key=lambda c: (c[2].get("wall_time") or 0.0, c[1]))
            pipe_chosen = pipe_tied[0]
            # Prefer the pipeline belonging to the winning trial itself.
            winner_params = _comparable_params(meta.get("params"))
            for candidate in pipe_tied:
                if (candidate[0] == score
                        and _comparable_params(candidate[2].get("params")) == winner_params):
                    pipe_chosen = candidate
                    break
            pipeline_score, _, pipe_meta = pipe_chosen
            pipeline_params = pipe_meta.get("params")
            # Its own threshold, so a fallback pipeline is stamped with the
            # threshold that actually belongs to it rather than the winner's.
            pipeline_th = pipe_meta.get("th")
            pipeline_is_winner = (
                pipeline_score == score
                and _comparable_params(pipeline_params) == winner_params
            )
            pipeline = self._load_blob(pipe_meta.get("blob"))

        return TrialRecord(
            score=score,
            th=meta.get("th"),
            th_to_rul=meta.get("th_to_rul"),
            pipeline=pipeline,
            pipeline_error=meta.get("pipeline_error"),
            params=meta.get("params"),
            matched_best_params=matched,
            tied_count=len(tied),
            pipeline_is_winner=pipeline_is_winner,
            pipeline_score=pipeline_score,
            pipeline_params=pipeline_params,
            pipeline_th=pipeline_th,
        )

    def is_better(self, candidate, incumbent):
        """Public direction-aware comparison (used by the experiment cross-check)."""
        return self._is_better(candidate, incumbent)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _is_better(self, candidate, incumbent):
        return candidate > incumbent if self.maximize else candidate < incumbent

    def _read_family(self, pattern):
        """Load one metadata family as ``[(score, filename, meta), ...]``.

        Unreadable or non-finite records are skipped rather than failing the
        whole recovery -- a partial result beats none.
        """
        found = []
        for path in sorted(glob.glob(os.path.join(self.spool_dir, pattern))):
            try:
                with open(path, "rb") as fh:
                    meta = json.loads(fh.read().decode("utf-8"))
                score = float(meta["score"])
            except Exception as exc:
                _log.warning(
                    "TrialSink: ignoring unreadable spool record %s (%s)", path, exc
                )
                continue
            if math.isfinite(score):
                found.append((score, os.path.basename(path), meta))
        return found

    def _load_blob(self, blob_name):
        """Unpickle one payload, or return ``None`` if it cannot be read."""
        if not blob_name:
            return None
        blob_path = os.path.join(self.spool_dir, blob_name)
        try:
            # cloudpickle streams are plain pickle streams; load with stdlib.
            with open(blob_path, "rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            _log.warning(
                "TrialSink: best-trial blob %s could not be loaded: %s", blob_path, exc
            )
            return None

    def _atomic_write(self, name, writer):
        """Write via a temp file in the *same* directory, then ``os.replace``.

        Same-directory temp guarantees the rename cannot cross a filesystem
        (``EXDEV``).  ``os.replace`` is atomic on POSIX and on Windows.  No
        ``fsync``: we defend against a killed worker (page cache survives), not
        against a machine crash (which ends the run anyway).
        """
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=self.spool_dir)
        try:
            with os.fdopen(fd, "wb") as fh:
                writer(fh)
            os.replace(tmp_path, os.path.join(self.spool_dir, name))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
