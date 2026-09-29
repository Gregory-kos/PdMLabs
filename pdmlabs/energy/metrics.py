"""Lifecycle energy metrics, adapted from E-LQO (SIGMOD 2027) to PdM.

Two layers, deliberately separated:

**Layer 1 -- measured, assumption-free.** Phase shares, optimizer overhead,
exploration waste, energy-per-quality-point, energy-to-target, and the
diminishing-returns fit. Everything here is computable from the exported
records alone and involves no judgement about what a failure costs.

**Layer 2 -- payoff-dependent.** :func:`eroi` and :func:`ept` need a *payoff*
per deployment cycle, which in PdM means converting detection quality into
avoided maintenance energy or cost. That conversion lives in
:mod:`pdmlabs.energy.payoff`; this module only does the amortisation algebra,
so the same functions serve the energy-denominated and cost-denominated models.

Why the adaptation is not a straight port
------------------------------------------
E-LQO's payoff is *more energy saved*: a better query plan executes more
cheaply, so the return on inference energy is itself energy. A better-tuned
anomaly detector does not make inference cheaper -- it makes *maintenance*
cheaper. There is therefore no ``E_exec`` analogue, and the numerator of EROI
has to come from outside the energy measurement. Everything downstream of that
substitution keeps E-LQO's algebra exactly, including its boundary cases.

Vocabulary
----------
``E_search``      HPO over the train split: all trials plus optimizer overhead.
                  The analogue of E-LQO's ``E_collect``, and expected to dominate.
``E_final_fit``   Refit of the winning configuration (``E_train``).
``E_infer``       Scoring the test split with the winning configuration.
``debt``          ``E_search + E_final_fit`` -- the one-time investment.
``cycle``         One pass over the test split, e.g. one month of fleet operation.
"""

from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# Layer 1: measured
# --------------------------------------------------------------------------- #

def optimizer_overhead(search_j, trial_js):
    """Energy the optimizer itself spent, outside any trial.

    ``E_opt = E_search - sum(E_trial)``: surrogate fitting, acquisition
    optimisation, space conversion and backend bookkeeping. This has no E-LQO
    analogue and is the natural question to ask of a *cross-optimizer* study --
    does a smarter search cost more in the optimizer than it saves in trials?

    Sound only when the phase envelope strictly enclosed every trial, which is
    why a negative residual is surfaced rather than clamped.
    """
    total = sum(t for t in trial_js if t is not None)
    e_opt = (search_j or 0.0) - total
    return {
        "e_optimizer_j": e_opt,
        "e_optimizer_frac": (e_opt / search_j) if search_j else None,
        "sum_trial_j": total,
        "search_j": search_j,
        "valid": e_opt >= -1e-6,
    }


def _incumbent_series(trials, maximize=True):
    """Running best-so-far over trials in execution order."""
    best = None
    out = []
    for t in trials:
        v = t.get("objective_value")
        if v is not None and (best is None or (v > best if maximize else v < best)):
            best = v
        out.append(best)
    return out


def exploration_waste(trials, maximize=True):
    """Energy spent on trials that never improved the incumbent.

    The direct analogue of E-LQO's finding that data collection -- executing
    plans that were mostly discarded -- dominates lifecycle energy. Cached
    trials are excluded: they did no work, so counting them would understate
    the waste fraction.
    """
    live = [t for t in trials if not t.get("cached")]
    inc = _incumbent_series(live, maximize)
    wasted = 0.0
    improving = 0
    prev = None
    for t, b in zip(live, inc):
        if prev is not None and b == prev:
            wasted += t.get("total_j") or 0.0
        else:
            improving += 1
        prev = b
    total = sum(t.get("total_j") or 0.0 for t in live)
    return {
        "e_wasted_j": wasted,
        "waste_frac": (wasted / total) if total else None,
        "n_trials": len(live),
        "n_improving": improving,
        "e_total_j": total,
    }


def convergence_trace(trials, maximize=True, energy_field="total_j"):
    """Cumulative search energy against incumbent quality.

    The input to the convergence-vs-energy plot, which is the fairest
    cross-optimizer comparator available: it asks how much energy each
    optimizer needed to reach a given quality, rather than how good it got
    given a fixed trial count. Trial *count* is not comparable across backends
    whose trials cost different amounts.
    """
    live = [t for t in trials if not t.get("cached")]
    inc = _incumbent_series(live, maximize)
    cum = 0.0
    trace = []
    for i, (t, b) in enumerate(zip(live, inc)):
        cum += t.get(energy_field) or 0.0
        trace.append({"trial_index": i, "cum_energy_j": cum,
                      "objective_value": t.get("objective_value"), "incumbent": b})
    return trace


def energy_to_target(trace, fractions=(0.9, 0.95, 0.99, 1.0), maximize=True):
    """Cumulative energy at which the incumbent first reached a quality target.

    Targets are fractions of the best value this run actually achieved, so the
    measure is self-normalising and comparable across datasets with different
    attainable quality.
    """
    if not trace:
        return {}
    finals = [p["incumbent"] for p in trace if p["incumbent"] is not None]
    if not finals:
        return {}
    best = max(finals) if maximize else min(finals)
    out = {}
    for f in fractions:
        target = best * f if maximize else best / f if best else best
        hit = None
        for p in trace:
            v = p["incumbent"]
            if v is None:
                continue
            if (v >= target) if maximize else (v <= target):
                hit = p["cum_energy_j"]
                break
        out["e_to_%g" % f] = hit
    out["best_objective"] = best
    return out


def energy_per_quality_point(search_j, q_tuned, q_baseline):
    """Joules of search bought each unit of quality over a baseline.

    ``None`` when the tuned run did not beat the baseline: dividing by a
    non-positive gain would produce a number that is not just meaningless but
    actively misleading in a table.
    """
    if q_tuned is None or q_baseline is None:
        return None
    gain = q_tuned - q_baseline
    if gain <= 0:
        return None
    return (search_j or 0.0) / gain


def fit_diminishing_returns(trace, maximize=True):
    """Fit ``Q(e) ~= Q* - beta / (1 + gamma * e)`` over normalised energy.

    E-LQO's Eq. 12, with exploration coverage replaced by normalised cumulative
    search energy. The useful output is the half-saturation point
    ``e_half = 1 / gamma``: the fraction of the energy budget at which half the
    attainable quality gain has been captured, i.e. where to stop searching.

    Falls back to a coarse empirical estimate when SciPy is unavailable or the
    fit does not converge, flagged by ``method``.
    """
    pts = [(p["cum_energy_j"], p["incumbent"]) for p in trace
           if p["incumbent"] is not None and p["cum_energy_j"] is not None]
    if len(pts) < 3:
        return {"method": "insufficient_data", "n_points": len(pts)}
    emax = max(e for e, _ in pts) or 1.0
    xs = [e / emax for e, _ in pts]
    ys = [q for _, q in pts]
    if not maximize:
        ys = [-y for y in ys]
    if max(ys) - min(ys) <= 0:
        return {"method": "no_variation", "n_points": len(pts),
                "note": "incumbent never improved; no curve to fit"}
    try:
        import numpy as np
        from scipy.optimize import curve_fit

        def model(x, q_star, beta, gamma):
            return q_star - beta / (1.0 + gamma * x)

        p0 = [max(ys), max(ys) - min(ys), 5.0]
        popt, _ = curve_fit(model, np.array(xs), np.array(ys), p0=p0, maxfev=20000)
        q_star, beta, gamma = (float(v) for v in popt)
        resid = [y - model(x, *popt) for x, y in zip(xs, ys)]
        ss_res = sum(r * r for r in resid)
        mean_y = sum(ys) / len(ys)
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        half = (1.0 / gamma) if gamma else None
        # e_half is only meaningful inside the observed budget. A tiny gamma
        # means the curve never flattened, so 1/gamma lands far outside the
        # data and would be reported as e.g. "half-saturation at 460802% of the
        # budget" -- an extrapolation dressed up as a measurement. A huge gamma
        # is the mirror case (saturated before the first point). Both are
        # reported as a regime, not a number.
        saturation = "within_budget"
        if half is None or half <= 0:
            saturation, half = "degenerate", None
        elif half > 1.0:
            saturation = "not_saturated_within_budget"
        elif half < 1e-3:
            saturation = "saturated_immediately"
        return {
            "method": "curve_fit",
            "q_star": q_star if maximize else -q_star,
            "beta": beta, "gamma": gamma,
            "saturation": saturation,
            "e_half_frac": half if saturation == "within_budget" else None,
            "e_half_frac_raw": half,
            "e_half_j": (emax * half) if (half is not None and saturation == "within_budget") else None,
            "e_total_j": emax, "n_points": len(pts),
            "r_squared": (1 - ss_res / ss_tot) if ss_tot else None,
        }
    except Exception as exc:
        span = max(ys) - min(ys)
        half = min(ys) + span / 2.0
        e_half = next((x for x, y in zip(xs, ys) if y >= half), None)
        return {"method": "empirical_fallback", "reason": str(exc)[:120],
                "e_half_frac": e_half,
                "e_half_j": (e_half * emax) if e_half is not None else None,
                "e_total_j": emax, "n_points": len(pts)}


# --------------------------------------------------------------------------- #
# Layer 2: amortisation (E-LQO Eq. 4, 5, 13, 14)
# --------------------------------------------------------------------------- #

def eroi(delta_payoff, e_infer):
    """Energy Return on Investment -- E-LQO Eq. 4.

    ``(payoff per cycle) / (online overhead per cycle)``. Above 1 means a cycle
    returns more than the inference it cost. Negative when the tuned pipeline is
    worse than the baseline, which is reported rather than hidden.
    """
    if e_infer in (None, 0) or delta_payoff is None:
        return None
    return delta_payoff / e_infer


def ept(debt, delta_payoff, e_infer):
    """Energy Payback Time in cycles -- E-LQO Eq. 5.

    ``min{k in Z+ : k * (payoff - e_infer) >= debt}``.

    Returns ``None`` for the two no-payback regimes E-LQO separates out, and
    says which one via :func:`payback_regime`. They must never be averaged into
    an aggregate: a mean EPT computed over runs where some never pay back is not
    a payback time.
    """
    if debt is None or delta_payoff is None or e_infer is None:
        return None
    net = delta_payoff - e_infer
    if net <= 0:
        return None
    if debt <= 0:
        return 1
    return int(math.ceil(debt / net))


def ept_from_eroi(debt, delta_payoff, eroi_value):
    """EPT via E-LQO Eq. 13, as an independent check on :func:`ept`.

    ``ceil(debt / (payoff * (1 - 1/EROI)))``. Algebraically identical to Eq. 5
    because ``payoff * (1 - e_infer/payoff) == payoff - e_infer``; implemented
    separately so the two can be asserted equal in tests.
    """
    if debt is None or delta_payoff is None or not eroi_value:
        return None
    factor = delta_payoff * (1.0 - 1.0 / eroi_value)
    if factor <= 0:
        return None
    if debt <= 0:
        return 1
    return int(math.ceil(debt / factor))


def payback_regime(delta_payoff, e_infer):
    """Name the regime, so undefined EPT is explained rather than blank."""
    if delta_payoff is None or e_infer is None:
        return "unknown"
    if delta_payoff < 0:
        return "worse_than_baseline"
    if delta_payoff == 0:
        return "no_benefit"
    if delta_payoff <= e_infer:
        return "benefit_below_inference_cost"
    return "pays_back"


def ept_effective(ept_value, rho):
    """EPT under distribution shift -- E-LQO Eq. 14, ``EPT_eff ~= EPT / rho^2``.

    Quadratic because similarity degrades both per-cycle savings and the
    accuracy of the predictions that produce them. Drift is more natural in PdM
    than in query workloads: sensors age and equipment is replaced.
    """
    if ept_value is None or not rho or rho <= 0:
        return None
    return int(math.ceil(ept_value / (rho * rho)))


def lifecycle_summary(search_j, final_fit_j, infer_j, baseline_infer_j=None,
                      delta_payoff=None, expected_cycles=None):
    """Assemble the lifecycle picture, including phase shares and payback."""
    debt = (search_j or 0.0) + (final_fit_j or 0.0)
    phases = {"E_search": search_j or 0.0, "E_final_fit": final_fit_j or 0.0,
              "E_infer": infer_j or 0.0}
    total = sum(phases.values())
    out = {
        "phases_j": phases,
        "phase_shares": {k: (v / total if total else None) for k, v in phases.items()},
        "total_j": total,
        "debt_j": debt,
        "baseline_infer_j": baseline_infer_j,
        "delta_payoff": delta_payoff,
        "regime": payback_regime(delta_payoff, infer_j),
    }
    r = eroi(delta_payoff, infer_j)
    out["eroi"] = r
    out["ept"] = ept(debt, delta_payoff, infer_j)
    out["ept_check_eq13"] = ept_from_eroi(debt, delta_payoff, r)
    if expected_cycles is not None and out["ept"] is not None:
        out["beneficial"] = expected_cycles > out["ept"]
        out["expected_cycles"] = expected_cycles
    return out
