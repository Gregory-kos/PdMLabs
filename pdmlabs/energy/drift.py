"""Distribution shift, for the effective-payback correction.

E-LQO's Eq. 14 states ``EPT_eff ~= EPT / rho^2``, where ``rho`` is the
similarity between the training and deployment distributions. The dependence is
quadratic because shift degrades both the per-cycle saving *and* the accuracy of
the predictions producing it.

The paper supplies the correction but not a way to measure ``rho``. This module
provides two estimators, because they answer different questions and can
disagree informatively:

* :func:`distribution_similarity` -- a *covariate* measure computed from the
  feature distributions alone. Available before any model is trained, and
  independent of how well the model happens to do.
* :func:`performance_similarity` -- an *outcome* measure: how much of the
  training-split quality survives on the deployment split. Directly reflects
  what shift costs, but conflates shift with ordinary generalisation gap.

Drift matters more in predictive maintenance than in query workloads: sensors
age, equipment is replaced, and operating regimes change seasonally, so a model
tuned on one period routinely meets a different one.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


def _numeric_columns(dfs):
    import pandas as pd
    cols = None
    for d in dfs:
        c = set(d.select_dtypes(include="number").columns)
        cols = c if cols is None else (cols & c)
    return sorted(cols or [])


def distribution_similarity(train_dfs, test_dfs, bins=30, columns=None):
    """Covariate similarity in ``[0, 1]`` via mean histogram intersection.

    For each numeric feature, both splits are histogrammed over a shared range
    and normalised to densities; their intersection (``sum(min(p, q))``) is the
    per-feature similarity, and the mean across features is ``rho``.

    Histogram intersection rather than a distance: it is bounded in ``[0, 1]``
    by construction, which is what Eq. 14 requires, needs no scale calibration,
    and degrades gracefully. An unbounded distance would have to be squashed by
    some arbitrary transform to be usable as ``rho``.

    Returns ``{'rho', 'per_feature', 'n_features'}``.
    """
    import numpy as np
    import pandas as pd

    train_dfs = [train_dfs] if isinstance(train_dfs, pd.DataFrame) else list(train_dfs)
    test_dfs = [test_dfs] if isinstance(test_dfs, pd.DataFrame) else list(test_dfs)
    if not train_dfs or not test_dfs:
        return {"rho": None, "per_feature": {}, "n_features": 0}

    a = pd.concat(train_dfs, ignore_index=True)
    b = pd.concat(test_dfs, ignore_index=True)
    cols = columns or _numeric_columns([a, b])
    if not cols:
        _log.warning("No shared numeric columns; rho is undefined.")
        return {"rho": None, "per_feature": {}, "n_features": 0}

    per = {}
    for c in cols:
        x = a[c].dropna().to_numpy(dtype=float)
        y = b[c].dropna().to_numpy(dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        if hi <= lo:
            per[c] = 1.0          # constant and identical in both splits
            continue
        hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
        hy, _ = np.histogram(y, bins=bins, range=(lo, hi), density=False)
        px = hx / hx.sum() if hx.sum() else hx
        py = hy / hy.sum() if hy.sum() else hy
        per[c] = float(np.minimum(px, py).sum())
    if not per:
        return {"rho": None, "per_feature": {}, "n_features": 0}
    rho = float(sum(per.values()) / len(per))
    return {"rho": rho, "per_feature": per, "n_features": len(per)}


def performance_similarity(q_train, q_test, maximize=True):
    """Outcome similarity: the fraction of tuned quality that survives deployment.

    Clipped to ``(0, 1]``. Values above 1 (the deployment split scoring better
    than the tuning split) are clipped to 1 rather than reported as
    "better than no shift": Eq. 14 has no meaning for ``rho > 1``, and the
    honest reading is simply that no degradation was observed.
    """
    if q_train is None or q_test is None:
        return None
    try:
        q_train, q_test = float(q_train), float(q_test)
    except (TypeError, ValueError):
        return None
    if q_train == 0:
        return None
    r = (q_test / q_train) if maximize else (q_train / q_test)
    if r <= 0:
        return None
    return min(1.0, r)


def effective_payback(ept_value, rho):
    """``EPT_eff = ceil(EPT / rho^2)`` with the inputs echoed back.

    Thin wrapper over :func:`pdmlabs.energy.metrics.ept_effective` that keeps
    ``rho`` and the inflation factor alongside the result, so a reported
    ``EPT_eff`` always carries the assumption that produced it.
    """
    from pdmlabs.energy import metrics as M
    eff = M.ept_effective(ept_value, rho)
    return {
        "ept": ept_value,
        "rho": rho,
        "ept_effective": eff,
        "inflation": (eff / ept_value) if (eff and ept_value) else None,
    }
