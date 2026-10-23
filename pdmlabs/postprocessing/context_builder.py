"""Build the context of every anomaly score with the PdmContext library (TECAD).

For each score of a target, the context gathers, over the time window that ends at the score,
the extra data of the target (events, external signals) and optionally some of its monitored
sensors, turned into series aligned with the scores, together with the relations that causal
discovery finds between those series and the score, their characterization and the resulting
interpretation of the score (Giannoulidis & Gounaris, "Explainable contextualized anomaly detection
for heterogeneous time series", JIIS 2026).

The contexts are built with ``PdmContext.ContextGeneration`` (``pip install "pdmlabs[context]"``):
`build_contexts` returns them all, `iter_contexts` yields them one at a time so that long targets
do not need memory for every window at once. Two behaviours of PdmContext 0.2.1 that would give
wrong results are worked around here, so the workarounds stay harmless if a later 0.2.x release
fixes them:

- A context returned by PdmContext shares lists with the library's internal state, and the next
  step appends to them: an already returned context would later end with the value of the *next*
  timestamp. Every context is therefore copied into a :class:`ScoreContext` as soon as it is
  returned, so no context ever holds information from after its own timestamp.
- PdmContext matches series names by substring: a series named ``pressure`` would also collect
  the samples of ``oil_pressure``, and a series whose name contains the score's (``score_raw``)
  would corrupt the score series itself. Every series is therefore handed to the library under a
  code that is never contained in another one (``[0]``, ``[1]``, ...), and translated back
  afterwards.
"""
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Name under which the anomaly score appears in every :class:`ScoreContext`.
SCORE = 'score'

#: Accepted values of the extra data's ``type`` column, mapped to PdmContext's type names.
CONTEXT_TYPES = {'isolated': 'isolated', 'configuration': 'configuration',
                 'univariate': 'Univariate', 'categorical': 'categorical'}

_HORIZON = re.compile(r'^\d+( (days|hours|minutes|seconds))?$')
_SOURCE = 'pdmlabs'
_SCORE_CODE = '[score]'
_CODE = re.compile(r'\[(\d+)\]')


@dataclass(frozen=True)
class ScoreContext:
    """The context of one anomaly score."""

    #: Time of the score.
    timestamp: pd.Timestamp
    #: Every series of the context window, aligned with the scores of the window, keyed by name:
    #: ``'score'`` for the anomaly scores, the extra data and monitored column names otherwise. A
    #: categorical series ``name`` appears as one indicator series ``'<category>_<name>'`` per
    #: category plus ``'state_<name>'``. A series with no sample in the window, or whose mapped
    #: values are all zero, is None; a category that does not occur in the window has no indicator
    #: series at all. Empty when the contexts were built with ``keep='last'``.
    series: dict
    #: For the same names as `series`, the value of each series at the score's own time (None
    #: where the series is None). Always present.
    last_values: dict
    #: Relations found by causal discovery between the series of the window, as PdmContext
    #: reports them: undirected, so every pair appears in both directions. Empty when the window
    #: holds five scores or fewer, or with ``causality='none'``.
    edges: tuple
    #: For each edge ``(a, b)``, in the same order, how PdmContext characterizes the influence of
    #: ``a`` on ``b``: ``'increase'``, ``'decrease'`` or ``'unknown'``.
    characterization: tuple
    #: PdmContext's interpretation of the score: the series it takes to explain the score at this
    #: time, as ``(series, 'score', characterization, since)`` tuples, where ``since`` is the
    #: time from which consecutive contexts gave that interpretation; ordered by ``since``.
    interpretation: tuple
    #: Only with ``strengths=True``, otherwise empty. For every series but the score, its partial
    #: correlation with the score over the window, given all the other series: in [-1, 1],
    #: positive when the score rises with it, near 0 when the two are unrelated once the others
    #: are accounted for. Where an edge only says whether causal discovery's test passed its
    #: threshold, this keeps the size and sign of the association. None when it is undefined: the
    #: series is None or constant in the window, it is fully determined by the other series (e.g.
    #: the indicators of a categorical series that sum to one), or the window has five scores or
    #: fewer.
    strengths: dict


def _no_causality(names, data, timestamps):
    """Causal discovery that finds nothing (PdmContext's own empty_cause takes two arguments)."""
    return []


def _causality_function(causality):
    if causality == 'none':
        return _no_causality
    if causality == 'pc':
        try:
            import castle  # noqa: F401  (gcastle, which PdmContext's PC relies on)
            from PdmContext.utils.causal_discovery_functions import calculate_with_pc
        except ImportError as error:
            raise ImportError(
                "causality='pc' needs PdmContext and gcastle; install them with: "
                "pip install \"pdmlabs[context]\""
            ) from error
        return calculate_with_pc
    raise ValueError(f"causality must be 'pc' or 'none', got {causality!r}.")


def _context_generator():
    try:
        from PdmContext.ContextGeneration import ContextGenerator
    except ImportError as error:
        raise ImportError(
            "Building contexts needs the PdmContext library; install it with: "
            "pip install \"pdmlabs[context]\""
        ) from error
    return ContextGenerator


def _same_timezone(index, times, what):
    """Return `index` in the timezone of `times`, or raise when only one of them has one."""
    index_aware, times_aware = index.tz is not None, times.tz is not None
    if index_aware != times_aware:
        raise ValueError(
            f"The {what} index is timezone-{'aware' if index_aware else 'naive'} but the score "
            f"times are timezone-{'aware' if times_aware else 'naive'}; make both consistent "
            "(for example with .tz_localize(None))."
        )
    return index.tz_convert(times.tz) if times_aware else index


def _extra_records(extra_data, times):
    """(timestamp, name, value, PdmContext type) for every row of the extra data."""
    if not isinstance(extra_data, pd.DataFrame):
        raise ValueError(f"extra_data must be a pandas DataFrame, got {type(extra_data).__name__}.")
    missing = [column for column in ('name', 'type') if column not in extra_data.columns]
    if missing:
        raise ValueError(f"extra_data must have the columns 'name' and 'type' (missing {missing}), "
                         f"got {list(extra_data.columns)}.")
    if not isinstance(extra_data.index, pd.DatetimeIndex):
        raise ValueError("extra_data must be indexed by time (a DatetimeIndex), as the "
                         "post-processor receives it.")
    index = _same_timezone(extra_data.index, times, 'extra_data')
    values = extra_data['value'] if 'value' in extra_data.columns else \
        pd.Series([None] * len(extra_data), index=extra_data.index)

    records = []
    for timestamp, name, kind, value in zip(index, extra_data['name'], extra_data['type'], values):
        if pd.isna(timestamp):
            continue
        context_type = CONTEXT_TYPES.get(str(kind).lower())
        if context_type is None:
            raise ValueError(f"Unknown extra_data type {kind!r} for {name!r}; expected one of "
                             f"{sorted(CONTEXT_TYPES)}.")
        if context_type in ('isolated', 'configuration'):
            value = None                         # events carry no value
        elif value is None or pd.isna(value):
            raise ValueError(f"extra_data rows of type {kind!r} need a value ({name!r} at "
                             f"{timestamp} has none).")
        elif context_type == 'Univariate':
            value = _number(value, f"extra_data series {name!r} (type {kind!r}) at {timestamp}")
        elif _CODE.search(str(value)):
            # The library names a category's series '<value>_<series code>', so a value that
            # looks like a code could be read back as another series.
            raise ValueError(f"Categorical value {value!r} of {name!r} contains '[<digits>]', "
                             "which context series names use internally; rename that category.")
        records.append((timestamp, str(name), value, context_type))
    return records


def _number(value, what):
    """`value` as a float, or a ValueError saying which series and time it belongs to."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"The value of {what} must be a number, got {value!r}.") from None


def _check_scores_and_times(scores, times):
    """Return `scores` as floats and `times` as a DatetimeIndex, or raise on what PdmContext cannot
    take: scores that are not finite numbers, times given as numbers, missing or repeated times."""
    values = []
    for position, score in enumerate(scores):
        if np.ndim(score) != 0:
            raise ValueError("scores must be numbers, one per time, got an array at position "
                             f"{position} (e.g. the survival curves of the SA flavor).")
        value = _number(score, f"the score at position {position}")
        if not np.isfinite(value):
            raise ValueError(f"scores must be finite, got {value} at position {position}.")
        values.append(value)
    raw = pd.Index(times)
    if pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw):
        # pandas would read them as nanoseconds since 1970, so any horizon would span them all.
        raise ValueError("times holds numbers, whose unit is ambiguous; convert them to datetimes "
                         "first, e.g. with pd.to_datetime(values, unit='s').")
    times = pd.DatetimeIndex(raw)
    if len(times) != len(values):
        raise ValueError(f"times has {len(times)} entries for {len(values)} scores.")
    if times.hasnans:
        raise ValueError("times has missing values.")
    if not times.is_monotonic_increasing:
        raise ValueError("times must be in increasing order.")
    if not times.is_unique:
        raise ValueError(f"times must not repeat ({times[times.duplicated()][0]} appears more than "
                         "once): PdmContext cannot place two scores at the same time.")
    return values, times


def _monitored_records(target_data, monitored_columns, times):
    """(timestamp, column, value, 'Univariate') for every non-missing monitored value."""
    if target_data is None:
        raise ValueError("monitored_columns needs target_data.")
    missing = [column for column in monitored_columns if column not in target_data.columns]
    if missing:
        raise ValueError(f"monitored_columns {missing} are not columns of target_data.")
    if len(target_data) != len(times) or not pd.DatetimeIndex(target_data.index).equals(times):
        raise ValueError("target_data must be indexed by the score times, one row per score.")
    records = []
    for column in monitored_columns:
        for timestamp, value in zip(times, target_data[column]):
            if not pd.isna(value):
                records.append((timestamp, str(column),
                                _number(value, f"monitored column {column!r} at {timestamp}"),
                                'Univariate'))
    return records


def iter_contexts(scores, times, extra_data=None, target_data=None, monitored_columns=None,
                  context_horizon='8 hours', causality='pc', keep='window', strengths=False):
    """Yield the context of every score of one target, one at a time.

    Same arguments and contexts as `build_contexts`, but nothing is accumulated: a consumer that
    handles each context and lets it go needs memory for one context at a time instead of all of
    them, which matters for long targets with long windows (e.g. 5-minute data with a one-day
    horizon). The inputs are checked when this function is called, before the first context.

    Args:
        keep (str): ``'window'`` to keep every series of each window in
            `ScoreContext.series`, or ``'last'`` to keep only `ScoreContext.last_values` (and the
            edges, characterization and interpretation), which is all a consumer of the latest
            values needs.
        strengths (bool): Also compute `ScoreContext.strengths`. It is computed on each full
            window as it is built, so it is available with ``keep='last'`` too.
        Other arguments: see `build_contexts`.

    Returns:
        Iterator[ScoreContext]: One context per score, in the same order.

    Raises:
        ImportError, ValueError: As `build_contexts`, raised by this call itself.
    """
    scores, times = _check_scores_and_times(list(scores), times)
    if not isinstance(context_horizon, str) or not _HORIZON.match(context_horizon):
        raise ValueError(f"context_horizon must look like '30 days' (a whole number of days, "
                         f"hours, minutes or seconds), got {context_horizon!r}.")
    if not isinstance(strengths, bool):
        raise ValueError(f"strengths must be True or False, got {strengths!r}.")
    if keep not in ('window', 'last'):
        raise ValueError(f"keep must be 'window' or 'last', got {keep!r}.")
    ContextGenerator = _context_generator()
    causality_function = _causality_function(causality)

    records = []
    if extra_data is not None:
        records += _extra_records(extra_data, times)
    if monitored_columns:
        records += _monitored_records(target_data, list(monitored_columns), times)

    names = list(dict.fromkeys(name for _, name, _, _ in records))
    if SCORE in names:
        raise ValueError(f"'{SCORE}' is reserved for the anomaly score; rename that series.")
    kinds = {}
    for _, name, _, kind in records:
        if kinds.setdefault(name, kind) != kind:
            raise ValueError(f"Series {name!r} appears with two types ({kinds[name]!r} and "
                             f"{kind!r}), e.g. as both extra data and a monitored column.")
    code_of = {name: f'[{i}]' for i, name in enumerate(names)}
    name_of = {code: name for name, code in code_of.items()}

    def decode(library_name):
        name = library_name[:-len(f'@{_SOURCE}')] if library_name.endswith(f'@{_SOURCE}') \
            else library_name
        if name == _SCORE_CODE:
            return SCORE
        if name in name_of:
            return name_of[name]
        # A categorical series: '<category>_<code>' or 'state_<code>'. Only that final code is a
        # series name; the category before it is the user's text and stays as it is.
        match = re.fullmatch(r'(.*)_(\[\d+\])', name)
        if match and match.group(2) in name_of:
            return f'{match.group(1)}_{name_of[match.group(2)]}'
        return name

    # Stable sort: rows with the same time keep their order and all come before the score.
    records.sort(key=lambda record: record[0])
    generator = ContextGenerator(target=_SCORE_CODE, context_horizon=context_horizon,
                                 Causalityfunct=causality_function)
    return _generate(generator, scores, times, records, code_of, decode, keep, strengths)


def _collect(generator, **kwargs):
    """One PdmContext call, with its numeric warnings silenced for that call only.

    Constant series in a window make the correlations of causal discovery divide by zero. The
    silencing must not outlive the call: between two contexts the generator below is suspended
    inside the caller's code, whose own warnings must stay visible.
    """
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        return generator.collect_data(source=_SOURCE, **kwargs)


#: Fewest scores in a window for which strengths are computed (PdmContext's rule for edges).
_MIN_WINDOW = 6
#: Residual variance, relative to the series' own, below which a series counts as determined by
#: the others.
_DETERMINED = 1e-8


def _partial_correlations(series):
    """Partial correlation of every series with the score over the window, given the others.

    Computed from regression residuals rather than by inverting the correlation matrix, so that
    a series the other series fully determine gets None instead of an arbitrary number.
    """
    score = np.asarray(series[SCORE], dtype=float)
    strengths = {name: None for name in series if name != SCORE}
    if len(score) < _MIN_WINDOW or np.ptp(score) == 0:
        return strengths
    usable = {name: np.asarray(values, dtype=float) for name, values in series.items()
              if name != SCORE and values is not None and len(values) == len(score)
              and np.ptp(np.asarray(values, dtype=float)) > 0}
    names = list(usable)
    for name in names:
        others = [usable[other] for other in names if other != name]
        design = np.column_stack([np.ones(len(score))] + others)
        x = usable[name]
        x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        score_residual = score - design @ np.linalg.lstsq(design, score, rcond=None)[0]
        if x_residual.var() <= _DETERMINED * x.var() or \
                score_residual.var() <= _DETERMINED * score.var():
            continue
        strengths[name] = float(np.corrcoef(x_residual, score_residual)[0, 1])
    return strengths


def _generate(generator, scores, times, records, code_of, decode, keep, strengths):
    position = 0
    for time, score in zip(times, scores):
        while position < len(records) and records[position][0] <= time:
            timestamp, name, value, kind = records[position]
            _collect(generator, timestamp=timestamp, name=code_of[name], value=value, type=kind)
            position += 1
        context = _collect(generator, timestamp=time, name=_SCORE_CODE, value=float(score),
                           type='Univariate')
        # Copy now: PdmContext keeps appending to lists this context still refers to.
        window = {}
        for key, values in context.CD.items():
            name = decode(key)
            if name in window:
                raise ValueError(
                    f"Two context series are named {name!r}: a categorical series produces "
                    "'<category>_<name>' and 'state_<name>' series, and one of them has the name "
                    "of another series; rename one of the two.")
            window[name] = None if values is None else list(values)
        last_values = {name: (None if values is None else values[-1])
                       for name, values in window.items()}
        edges = tuple((decode(a), decode(b)) for a, b in context.CR['edges'])
        interpretation = tuple((decode(cause), decode(target), character, since)
                               for cause, target, character, since in
                               context.CR['interpretation'])
        yield ScoreContext(timestamp=time, series={} if keep == 'last' else window,
                           last_values=last_values, edges=edges,
                           characterization=tuple(context.CR['characterization']),
                           interpretation=interpretation,
                           strengths=_partial_correlations(window) if strengths else {})


def build_contexts(scores, times, extra_data=None, target_data=None, monitored_columns=None,
                   context_horizon='8 hours', causality='pc', keep='window', strengths=False):
    """Build the context of every score of one target.

    The data are fed to PdmContext in time order: before the score at time t, every extra-data
    and monitored row up to t (and at t) has been fed, and nothing later, so a context only
    holds information available at its own time. All contexts are returned at once; for long
    targets use `iter_contexts`, which yields them one at a time.

    Args:
        scores (list[float]): Anomaly scores of the target, in time order: finite numbers.
        times (pd.DatetimeIndex or sequence): Time of each score, increasing (no repeats), as
            datetimes, not numbers.
        extra_data (pd.DataFrame, optional): The target's extra data, indexed by time (as the
            post-processor's ``transform`` receives it), with the columns ``name``, ``type``
            (``isolated``, ``configuration``, ``univariate`` or ``categorical``) and, for the last
            two, ``value``.
        target_data (pd.DataFrame, optional): The target's data, one row per score, indexed by
            ``times``. Only read for `monitored_columns`.
        monitored_columns (list[str], optional): Columns of `target_data` to add to the context as
            continuous series. Every series is one more variable for causal discovery, whose cost
            grows quickly with their number, so keep this list short.
        context_horizon (str): Length of the context window, in PdmContext's format: a whole
            number followed by ``days``, ``hours``, ``minutes`` or ``seconds`` (``"30 days"``); a
            bare number means hours. A window holds the scores of the last horizon up to t
            (PdmContext 0.2.1 also keeps one or two older ones).
        causality (str): ``'pc'`` for PdmContext's PC algorithm (needs gcastle) or ``'none'`` to
            build the series only (no edges, characterization or interpretation).
        keep (str): ``'window'`` (default) keeps every series of each window; ``'last'`` keeps
            only the last value of each series, see `ScoreContext`.
        strengths (bool): Also compute the partial correlation of every series with the score
            (`ScoreContext.strengths`). Off by default: it adds regressions to every window.

    Returns:
        list[ScoreContext]: One context per score, in the same order.

    Raises:
        ImportError: When PdmContext (or gcastle, for ``causality='pc'``) is not installed.
        ValueError: On inputs that cannot be turned into contexts; the message says why.
    """
    return list(iter_contexts(scores, times, extra_data=extra_data, target_data=target_data,
                              monitored_columns=monitored_columns,
                              context_horizon=context_horizon, causality=causality, keep=keep,
                              strengths=strengths))
