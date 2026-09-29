"""Payoff models: turning detection quality into avoided energy or cost.

E-LQO's EROI and EPT both need a *payoff per deployment cycle*. In learned query
optimization that payoff is energy itself -- a better plan executes more
cheaply -- so the metric is self-contained. In predictive maintenance it is not:
a better-tuned detector does not make inference cheaper, it avoids unplanned
failures and unnecessary call-outs. The payoff therefore has to come from a
model of what those outcomes cost, and that model is an assumption.

This module makes the assumption explicit, configurable and testable rather than
burying it. Two providers, because some datasets ship a cost matrix (SCANIA
Component X) and some do not:

* :class:`EnergyPayoffModel` -- coefficients in joules. Use when there is no
  cost information but the physical cost of a failure can be reasoned about
  (lost production, corrective repair).
* :class:`CostPayoffModel` -- coefficients in currency, with an electricity
  price converting the compute side into the same units.

Both expose the same two methods, so :mod:`pdmlabs.energy.metrics` computes
EROI and EPT identically for either:

``payoff_per_cycle(tuned, baseline)``
    Benefit of the tuned pipeline over the baseline, per cycle, in model units.
``compute_cost(energy_j)``
    A compute energy expressed in those same model units.

Counting at the right granularity
---------------------------------
Costs are charged **per event, not per sample**. A cost matrix says "a missed
failure costs X" and "an unnecessary inspection costs Y" -- one sustained alarm
is one call-out, not one per timestamp. Charging point-level counts would
overstate inspection cost by orders of magnitude. The defaults therefore use
``fn_episodes`` and ``n_false_alarm_groups`` from
:func:`pdmlabs.evaluation.evaluation.outcome_counts`.

Because the coefficients are assumptions, never report a single EROI or EPT from
them without :func:`sensitivity` or :func:`breakeven_ratio` alongside.
"""

from __future__ import annotations

import abc
import math

from pdmlabs.energy.reading import KWH_TO_J


class BasePayoffModel(abc.ABC):
    """Converts outcome counts into a per-cycle benefit, in model units."""

    units = "unknown"

    @abc.abstractmethod
    def maintenance_cost(self, counts) -> float:
        """Cost of one cycle's maintenance outcomes, in model units."""

    @abc.abstractmethod
    def compute_cost(self, energy_j) -> float:
        """A compute energy in joules, expressed in model units."""

    def payoff_per_cycle(self, counts_tuned, counts_baseline):
        """Benefit of *tuned* over *baseline* for one cycle.

        Positive means the tuned pipeline avoided cost. Negative is returned as
        such, not clamped: EROI is defined as negative in that regime and
        E-LQO reports those cases separately rather than hiding them.
        """
        if counts_tuned is None or counts_baseline is None:
            return None
        return self.maintenance_cost(counts_baseline) - self.maintenance_cost(counts_tuned)


def _get(counts, key, default=0.0):
    v = counts.get(key, counts.get("count_" + key, default))
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class EnergyPayoffModel(BasePayoffModel):
    """Payoff denominated in joules.

    Parameters are the energy consequences of each outcome:

    e_fail
        An unplanned failure: lost production plus corrective repair. Dominates
        everything else, usually by orders of magnitude.
    e_inspect
        One unnecessary inspection triggered by a false alarm.
    e_planned
        One planned repair following a correct, timely detection -- the cost
        that is *incurred* in order to avoid ``e_fail``.
    """

    units = "J"

    def __init__(self, e_fail, e_inspect, e_planned=0.0, episode_level=True):
        self.e_fail = float(e_fail)
        self.e_inspect = float(e_inspect)
        self.e_planned = float(e_planned)
        self.episode_level = episode_level

    def maintenance_cost(self, counts):
        if self.episode_level:
            fn = _get(counts, "fn_episodes")
            fp = _get(counts, "n_false_alarm_groups")
            tp = _get(counts, "n_detected_episodes")
        else:
            fn, fp, tp = _get(counts, "fn"), _get(counts, "fp"), _get(counts, "tp")
        return fn * self.e_fail + fp * self.e_inspect + tp * self.e_planned

    def compute_cost(self, energy_j):
        return None if energy_j is None else float(energy_j)


class CostPayoffModel(BasePayoffModel):
    """Payoff denominated in currency, for datasets that ship a cost matrix.

    ``electricity_price_per_kwh`` converts compute energy into the same units,
    so EROI stays a pure ratio rather than mixing joules with currency.
    """

    units = "currency"

    def __init__(self, c_fail, c_inspect, c_planned=0.0,
                 electricity_price_per_kwh=0.0, episode_level=True):
        self.c_fail = float(c_fail)
        self.c_inspect = float(c_inspect)
        self.c_planned = float(c_planned)
        self.electricity_price_per_kwh = float(electricity_price_per_kwh)
        self.episode_level = episode_level

    def maintenance_cost(self, counts):
        if self.episode_level:
            fn = _get(counts, "fn_episodes")
            fp = _get(counts, "n_false_alarm_groups")
            tp = _get(counts, "n_detected_episodes")
        else:
            fn, fp, tp = _get(counts, "fn"), _get(counts, "fp"), _get(counts, "tp")
        return fn * self.c_fail + fp * self.c_inspect + tp * self.c_planned

    def compute_cost(self, energy_j):
        if energy_j is None:
            return None
        return (energy_j / KWH_TO_J) * self.electricity_price_per_kwh


#: The classic SCANIA APS cost matrix (UCI): 10 per unnecessary check, 500 per
#: missed failure. A useful sanity reference and a reasonable default when a
#: dataset ships no matrix of its own -- but SCANIA Component X publishes its
#: own, which should be preferred when working with that dataset.
SCANIA_APS_COSTS = {"c_inspect": 10.0, "c_fail": 500.0}


def scania_aps_model(electricity_price_per_kwh=0.0):
    """CostPayoffModel preset for the SCANIA APS cost matrix."""
    return CostPayoffModel(c_fail=SCANIA_APS_COSTS["c_fail"],
                           c_inspect=SCANIA_APS_COSTS["c_inspect"],
                           electricity_price_per_kwh=electricity_price_per_kwh)


# --------------------------------------------------------------------------- #
# Sensitivity -- mandatory, because the coefficients are assumptions
# --------------------------------------------------------------------------- #

def evaluate(model, counts_tuned, counts_baseline, debt_j, infer_j,
             expected_cycles=None):
    """EROI, EPT and regime for one payoff model."""
    from pdmlabs.energy import metrics as M

    payoff = model.payoff_per_cycle(counts_tuned, counts_baseline)
    infer_c = model.compute_cost(infer_j)
    debt_c = model.compute_cost(debt_j)
    r = M.eroi(payoff, infer_c)
    e = M.ept(debt_c, payoff, infer_c)
    out = {
        "units": model.units,
        "payoff_per_cycle": payoff,
        "infer_cost": infer_c,
        "debt_cost": debt_c,
        "eroi": r,
        "ept": e,
        "ept_check_eq13": M.ept_from_eroi(debt_c, payoff, r),
        "regime": M.payback_regime(payoff, infer_c),
    }
    if expected_cycles is not None:
        out["expected_cycles"] = expected_cycles
        out["beneficial"] = (e is not None and expected_cycles > e)
    return out


def sensitivity(model_factory, counts_tuned, counts_baseline, debt_j, infer_j,
                ratios):
    """Sweep the failure-to-inspection cost ratio.

    *model_factory* takes a ratio and returns a payoff model. Reporting a single
    EPT from one guessed coefficient pair would be false precision; the shape of
    EPT across plausible ratios is the honest result.
    """
    out = []
    for ratio in ratios:
        res = evaluate(model_factory(ratio), counts_tuned, counts_baseline,
                       debt_j, infer_j)
        res["ratio"] = ratio
        out.append(res)
    return out


def breakeven_ratio(model_factory, counts_tuned, counts_baseline, debt_j,
                    infer_j, max_cycles=None, lo=1.0, hi=1e6, tol=1e-3):
    """Smallest failure-to-inspection ratio at which tuning pays back.

    With *max_cycles* given, the smallest ratio whose EPT falls below it;
    otherwise the smallest ratio at which EPT is defined at all.

    This is the most assumption-light result available from Layer 2. Rather than
    asserting a payback time from coefficients nobody can pin down, it inverts
    the question into one a maintenance engineer can actually answer: *"tuning
    pays for itself provided a missed failure costs at least N inspections."*

    Bisection is valid because EPT is monotone non-increasing in the ratio: a
    costlier failure can only increase the benefit of catching it.
    """
    def ok(ratio):
        res = evaluate(model_factory(ratio), counts_tuned, counts_baseline,
                       debt_j, infer_j)
        if res["ept"] is None:
            return False
        return True if max_cycles is None else res["ept"] <= max_cycles

    if ok(lo):
        return lo
    if not ok(hi):
        return None
    while hi - lo > tol * max(1.0, lo):
        mid = math.sqrt(lo * hi) if lo > 0 else (lo + hi) / 2.0
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi
