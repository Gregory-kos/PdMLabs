"""Tests for `pdmlabs.postprocessing.context_builder` (needs ``pip install "pdmlabs[context]"``).

They check what the module guarantees on top of PdmContext: no context holds information from
after its own time (even after later contexts were built), series names are kept apart exactly,
the four kinds of extra data come through, and inputs PdmContext cannot take are rejected with a
clear message. Each `test_*` function can be collected by pytest; `main()` runs them all
(`python test/test_context_builder.py`). Without PdmContext the script only reports that it skipped.
"""
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

HAVE_PDMCONTEXT = importlib.util.find_spec('PdmContext') is not None

from pdmlabs.postprocessing.context_builder import SCORE, build_contexts

TIMES = pd.date_range('2024-01-01', periods=24, freq='h')
SCORES = list(np.sin(np.linspace(0, 6, 24)) + 2)


def _extra(rows):
    """rows: (position in TIMES, name, type, value)."""
    return pd.DataFrame({'name': [r[1] for r in rows], 'type': [r[2] for r in rows],
                         'value': [r[3] for r in rows]},
                        index=pd.DatetimeIndex([TIMES[r[0]] for r in rows]))


EXTRA = _extra([(3, 'alarm', 'isolated', None), (6, 'setting', 'configuration', None),
                (2, 'temperature', 'univariate', 60.0), (9, 'temperature', 'univariate', 65.0),
                (15, 'temperature', 'univariate', 70.0), (5, 'mode', 'categorical', 'run'),
                (12, 'mode', 'categorical', 'idle'), (20, 'alarm', 'isolated', None)])


def _raises(call, fragment):
    try:
        call()
    except ValueError as error:
        assert fragment in str(error), (fragment, str(error))
        return
    raise AssertionError(f'expected a ValueError containing {fragment!r}')


def test_no_context_holds_later_information():
    """Each context equals the one built from the data up to its own time only, so nothing from
    later rows reached it, not even through lists PdmContext keeps appending to."""
    full = build_contexts(SCORES, TIMES, extra_data=EXTRA, causality='none')
    for k in (4, 10, 16, 23):
        alone = build_contexts(SCORES[:k + 1], TIMES[:k + 1], extra_data=EXTRA[EXTRA.index <= TIMES[k]],
                               causality='none')[-1]
        assert full[k].series == alone.series, k
        assert full[k].last_values == alone.last_values, k


def test_series_names_are_kept_apart():
    """PdmContext matches names by substring; here every name keeps its own values."""
    extra = _extra([(20, 'pressure', 'univariate', 1.0), (21, 'oil_pressure', 'univariate', 9.0),
                    (22, 'score_raw', 'univariate', 5.0)])
    last = build_contexts(SCORES, TIMES, extra_data=extra, causality='none')[-1].last_values
    assert last['pressure'] == 1.0 and last['oil_pressure'] == 9.0 and last['score_raw'] == 5.0
    assert last[SCORE] == SCORES[-1]


def test_the_four_kinds_of_extra_data():
    # A one-day window, so that the last context's window holds every row of EXTRA.
    series = build_contexts(SCORES, TIMES, extra_data=EXTRA, causality='none',
                            context_horizon='24 hours')[-1].series
    for name in ('alarm', 'setting', 'temperature', 'idle_mode', 'state_mode'):
        assert name in series, (name, sorted(series))


def test_keep_last_and_strengths():
    window = build_contexts(SCORES, TIMES, extra_data=EXTRA, causality='none', strengths=True)
    last = build_contexts(SCORES, TIMES, extra_data=EXTRA, causality='none', keep='last',
                          strengths=True)
    for full, light in zip(window, last):
        assert light.series == {} and light.last_values == full.last_values
        assert light.strengths == full.strengths
        assert all(value is None or -1 <= value <= 1 for value in full.strengths.values())


def test_inputs_are_rejected_with_a_clear_message():
    scores, times = SCORES[:12], TIMES[:12]
    repeated = TIMES[:11].append(TIMES[10:11])
    _raises(lambda: build_contexts(scores, range(12), causality='none'), 'times holds numbers')
    _raises(lambda: build_contexts(scores, repeated, causality='none'), 'times must not repeat')
    _raises(lambda: build_contexts(scores[:5] + [float('nan')] + scores[6:], times,
                                   causality='none'), 'scores must be finite')
    _raises(lambda: build_contexts([np.array([0.9, 0.8])] * 12, times, causality='none'),
            'scores must be numbers')
    _raises(lambda: build_contexts(scores, times, extra_data=_extra(
        [(10, 'temperature', 'univariate', 'hot')]), causality='none'), 'must be a number')
    _raises(lambda: build_contexts(scores, times, extra_data=_extra(
        [(10, 'mode', 'categorical', 'm[7]')]), causality='none'), "contains '[<digits>]'")
    _raises(lambda: build_contexts(scores, times, extra_data=_extra(
        [(9, 'pump', 'categorical', 'on'), (10, 'state_pump', 'univariate', 3.0)]),
        causality='none'), "Two context series are named 'state_pump'")


def test_causality_function():
    """A causal discovery function of the user's sees the series names and its edges come back."""
    calls = []

    def drivers_of_the_score(names, data, timestamps):
        calls.append((list(names), np.shape(data), len(timestamps)))
        return np.array([(name, SCORE) for name in ('temperature', 'run_mode') if name in names])

    contexts = build_contexts(SCORES, TIMES, extra_data=EXTRA, context_horizon='24 hours',
                              causality=drivers_of_the_score)
    assert calls and all(shape == (rows, len(names)) for names, shape, rows in calls), calls[:3]
    # Every name the function sees is a real series name, the categorical ones included.
    assert set().union(*(names for names, _, _ in calls)) == {
        SCORE, 'temperature', 'alarm', 'setting', 'run_mode', 'idle_mode', 'state_mode'}
    assert contexts[-1].edges == (('temperature', SCORE), ('run_mode', SCORE))
    assert len(contexts[-1].characterization) == 2
    assert build_contexts(SCORES, TIMES, extra_data=EXTRA,
                          causality=lambda names, data, timestamps: None)[-1].edges == ()

    def build(edges):
        return lambda: build_contexts(SCORES, TIMES, extra_data=EXTRA, context_horizon='24 hours',
                                      causality=lambda names, data, timestamps: edges)
    for bad in ([('pump', SCORE)], [(SCORE, SCORE)], [('temperature', 'temperature')],
                [(['temperature'], SCORE)], [('temperature',)], ['temperature']):
        _raises(build(bad), 'every edge must be a pair of two different names')
    _raises(lambda: build_contexts(SCORES, TIMES, causality='granger'), "a function")

    def failing(names, data, timestamps):
        raise KeyError('mine')
    try:
        build_contexts(SCORES, TIMES, extra_data=EXTRA, causality=failing)
    except KeyError as error:
        assert 'window ending at' in ' '.join(getattr(error, '__notes__', []))
    else:
        raise AssertionError('the error of the causality function was lost')

    def warns(names, data, timestamps):
        warnings.warn('from the function', RuntimeWarning)
        np.float64(1.0) / np.float64(0.0)
        return []
    with warnings.catch_warnings(record=True) as caught, np.errstate(divide='warn'):
        warnings.simplefilter('always')
        build_contexts(SCORES, TIMES, extra_data=EXTRA, causality=warns)
    messages = {str(w.message) for w in caught}
    assert 'from the function' in messages, messages
    assert any('divide' in m for m in messages), messages


def test_pc_causality():
    if importlib.util.find_spec('castle') is None:
        print('   (gcastle not installed: PC causality not checked)')
        return
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        contexts = build_contexts(SCORES, TIMES, extra_data=EXTRA, causality='pc')
    assert len(contexts) == len(SCORES)
    for context in contexts:
        assert all(a in context.series and b in context.series for a, b in context.edges)
        assert len(context.characterization) == len(context.edges)
        assert set(context.characterization) <= {'increase', 'decrease', 'unknown'}
        for cause, target, character, since in context.interpretation:
            assert cause in context.series and target == SCORE, (cause, target)
            assert character in ('increase', 'decrease', 'unknown') and since <= context.timestamp
    assert any(context.interpretation for context in contexts)
    # The interpretation also depends on earlier contexts: still nothing from after a context.
    for k in (10, 23):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            alone = build_contexts(SCORES[:k + 1], TIMES[:k + 1],
                                   extra_data=EXTRA[EXTRA.index <= TIMES[k]], causality='pc')[-1]
        assert (contexts[k].edges, contexts[k].characterization, contexts[k].interpretation) == \
            (alone.edges, alone.characterization, alone.interpretation), k


def main():
    if not HAVE_PDMCONTEXT:
        print('Skipped: PdmContext is not installed (pip install "pdmlabs[context]").')
        return
    tests = [
        test_no_context_holds_later_information,
        test_series_names_are_kept_apart,
        test_the_four_kinds_of_extra_data,
        test_keep_last_and_strengths,
        test_inputs_are_rejected_with_a_clear_message,
        test_causality_function,
        test_pc_causality,
    ]
    for test in tests:
        test()
        print(f'OK: {test.__name__}')
    print('All context builder tests passed.')


if __name__ == '__main__':
    main()
