"""Tests for the post-processor `transform(scores, source, event_data, target_data, extra_data)` API.

The fast unit tests check `postprocess_scores` on its own. The flavor tests run every batch
experiment flavor on `data/ims.csv` with a recording post-processor and check that each
`transform` call receives its own target's data, before preprocessing, and its own target's extra
data; they take a few minutes. Each `test_*` function can be collected by pytest; `main()` runs
them all (`python test/test_postprocessor_api.py`). Run it from a scratch directory: the flavor
tests create MLflow runs in the current directory, like the other scripts in `test/`.
"""
import os
import sys
import warnings

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdmlabs.utils.dataset import Dataset
from pdmlabs.RunExperiment import run_experiment
from pdmlabs.postprocessing.post_processor import PostProcessorInterface, postprocess_scores
from pdmlabs.preprocessing.record_level.min_max_scaler import MinMaxScaler
from pdmlabs.preprocessing.record_level.aggregator import MeanAggregator
from pdmlabs.experiment.batch.semi_supervised_experiment import SemiSupervisedPdMExperiment
from pdmlabs.experiment.batch.supervised_experiment import SupervisedPdMExperiment
from pdmlabs.experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment
from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import \
    AutoProfileSemiSupervisedPdMExperiment
from pdmlabs.experiment.batch.incremental_semi_supervised_experiment import \
    IncrementalSemiSupervisedPdMExperiment
from pdmlabs.experiment.batch.RUL_experiment import SupervisedRULPdMExperiment
from pdmlabs.experiment.batch.SA_experiment import Supervised_SA_PdMExperiment
from pdmlabs.method.ocsvm import OneClassSVM
from pdmlabs.method.xgboost import XGBoost
from pdmlabs.method.xgboostRUL import XGBoostRUL
from pdmlabs.method.isolation_forest_uns import IsolationForestUnsupervised
from pdmlabs.method.CoxModel import CoxPH
from pdmlabs.thresholding.SurvSuperVisedTH import SurvToRUL

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ims.csv')
DATES = 'Artificial_timestamp'

#: Every transform call of RecordingPostProcessor, in order.
CALLS = []


class _MinimalPostProcessor(PostProcessorInterface):
    """Implements everything but `transform`, for the unit tests."""

    def fit(self, historic_data, historic_sources, event_data, anomaly_ranges=None):
        pass

    def transform_one(self, score_point, source, is_event):
        return score_point

    def get_params(self):
        return {}

    def __str__(self):
        return type(self).__name__


class LegacyPostProcessor(_MinimalPostProcessor):
    """Written for the older three-argument signature."""

    def transform(self, scores, source, event_data):
        return [score * 2 for score in scores]


class ContextPostProcessorStub(_MinimalPostProcessor):
    def transform(self, scores, source, event_data, target_data=None, extra_data=None):
        self.received = (target_data, extra_data)
        return scores


class KwargsPostProcessor(_MinimalPostProcessor):
    def transform(self, scores, source, event_data, **kwargs):
        self.received = kwargs
        return scores


class WrongLengthPostProcessor(_MinimalPostProcessor):
    def transform(self, scores, source, event_data, target_data=None, extra_data=None):
        return scores[:-1]


class RecordingPostProcessor(_MinimalPostProcessor):
    """Records what every `transform` call receives and returns the scores unchanged."""

    def __init__(self, event_preferences, **kwargs):
        super().__init__(event_preferences)

    def transform(self, scores, source, event_data, target_data=None, extra_data=None):
        CALLS.append({
            'n_scores': len(scores),
            'target_data': None if target_data is None else target_data.copy(),
            'extra_data': None if extra_data is None else extra_data.copy(),
        })
        return scores

    def __str__(self):
        return 'Recording'


class ExtraRowCountPostProcessor(_MinimalPostProcessor):
    """Adds the number of extra-data rows to every score, to see extra data reach a loaded model."""

    def __init__(self, event_preferences, **kwargs):
        super().__init__(event_preferences)

    def transform(self, scores, source, event_data, target_data=None, extra_data=None):
        shift = 0 if extra_data is None else len(extra_data)
        return [score + shift for score in scores]

    def __str__(self):
        return 'ExtraRowCount'


# --------------------------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------------------------- #
def test_legacy_signature_still_works():
    processed = postprocess_scores(LegacyPostProcessor({}), [1.0, 2.0], 'src', pd.DataFrame(),
                                   target_data=pd.DataFrame({'a': [1, 2]}),
                                   extra_data=pd.DataFrame({'b': [3]}))
    assert processed == [2.0, 4.0]


def test_new_signature_receives_target_and_extra_data():
    index = pd.date_range('2024-01-01', periods=3, freq='h')
    target = pd.DataFrame({'a': [1, 2, 3]}, index=index)
    extra = pd.DataFrame({'name': ['e']}, index=index[:1])
    for processor in (ContextPostProcessorStub({}), KwargsPostProcessor({})):
        postprocess_scores(processor, [0.1, 0.2, 0.3], 'src', pd.DataFrame(), target_data=target,
                           extra_data=extra)
        received = processor.received
        got_target, got_extra = (received if isinstance(received, tuple)
                                 else (received['target_data'], received['extra_data']))
        assert got_target.equals(target) and got_extra.equals(extra), type(processor).__name__


def test_wrong_number_of_scores_raises():
    try:
        postprocess_scores(WrongLengthPostProcessor({}), [0.1, 0.2], 'src', pd.DataFrame())
        assert False, "expected a ValueError"
    except ValueError:
        pass


def test_target_data_is_matched_to_the_scored_rows():
    index = pd.date_range('2024-01-01', periods=6, freq='h')
    target = pd.DataFrame({'a': range(6)}, index=index)
    processor = ContextPostProcessorStub({})

    # A preprocessor dropped the first two rows but kept the timestamps of the others.
    postprocess_scores(processor, [0.0] * 4, 'src', pd.DataFrame(), target_data=target,
                       scored_index=index[2:])
    assert processor.received[0]['a'].tolist() == [2, 3, 4, 5]

    # One row per score but new (resampled) labels: used as is, in order.
    postprocess_scores(processor, [0.0] * 6, 'src', pd.DataFrame(), target_data=target,
                       scored_index=index + pd.Timedelta('1min'))
    assert processor.received[0].equals(target)

    # Rows merged into fewer, new rows: no row-per-score correspondence, so None.
    postprocess_scores(processor, [0.0] * 3, 'src', pd.DataFrame(), target_data=target,
                       scored_index=pd.date_range('2024-01-01', periods=3, freq='2h') +
                       pd.Timedelta('1min'))
    assert processor.received[0] is None


def test_unmatched_target_data_warns_only_when_needed():
    """A post-processor that sets needs_target_data is warned when it gets None; others are not."""
    index = pd.date_range('2024-01-01', periods=6, freq='h')
    target = pd.DataFrame({'a': range(6)}, index=index)
    merged = pd.date_range('2024-01-01', periods=3, freq='2h') + pd.Timedelta('1min')

    class NeedsTargetData(ContextPostProcessorStub):
        needs_target_data = True

    for processor, scored_index, should_warn in (
            (NeedsTargetData({}), merged, True),            # unmatched and needed
            (ContextPostProcessorStub({}), merged, False),  # unmatched but not needed
            (NeedsTargetData({}), index[3:], False)):       # matched
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            postprocess_scores(processor, [0.0] * 3, 'src', pd.DataFrame(), target_data=target,
                               scored_index=scored_index)
        warned = any('receives target_data=None' in str(w.message) for w in caught)
        assert warned == should_warn, (type(processor).__name__, should_warn)


def test_merged_rows_give_no_target_data():
    """A preprocessor that merges rows (MeanAggregator) leaves no input row behind a score, even
    when its bin labels are timestamps of target_data: target_data is then None, with a warning
    for a post-processor that needs it, rather than the first row of each bin."""
    index = pd.date_range('2024-01-01', periods=60, freq='min')
    target = pd.DataFrame({'a': np.arange(60.0)}, index=index)
    aggregator = MeanAggregator(event_preferences={'failure': [], 'reset': []}, period='10min')
    aggregator.fit([target], ['src'], pd.DataFrame())
    aggregated = aggregator.transform(target, 'src', pd.DataFrame())
    assert aggregated.index.isin(index).all()          # the case that used to be mismatched

    class NeedsTargetData(ContextPostProcessorStub):
        needs_target_data = True

    processor = NeedsTargetData({})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        postprocess_scores(processor, [0.0] * len(aggregated), 'src', pd.DataFrame(),
                           target_data=target, scored_index=aggregated.index,
                           rows_merged=MeanAggregator.merges_rows)
    assert processor.received[0] is None
    assert any('merges rows' in str(w.message) for w in caught)

    # Nothing merged (one row per period already): the rows are passed as they are.
    hourly = target.iloc[::60]
    postprocess_scores(processor, [0.0], 'src', pd.DataFrame(), target_data=hourly,
                       scored_index=hourly.index, rows_merged=True)
    assert processor.received[0].equals(hourly)


def test_extra_data_must_be_indexed_by_time():
    """Extra data with rows that is not indexed by time is rejected for a transform that takes it,
    instead of reaching it with an integer index."""
    unindexed = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
                              'name': ['e', 'f']})
    try:
        postprocess_scores(ContextPostProcessorStub({}), [0.1], 'src', pd.DataFrame(),
                           extra_data=unindexed)
        assert False, "expected a ValueError"
    except ValueError as error:
        assert 'indexed by time' in str(error)
    # Accepted: no rows, or a transform that does not take extra_data at all.
    postprocess_scores(ContextPostProcessorStub({}), [0.1], 'src', pd.DataFrame(),
                       extra_data=unindexed.iloc[0:0])
    assert postprocess_scores(LegacyPostProcessor({}), [0.1], 'src', pd.DataFrame(),
                              extra_data=unindexed) == [0.2]


# --------------------------------------------------------------------------------------------- #
# Flavor tests
# --------------------------------------------------------------------------------------------- #
def _ims_dataset():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Dataset(data=pd.read_csv(DATA_PATH), datetime_column=DATES,
                       event_indicator='event', source_column='source', train_sources=['1'],
                       val_sources=['2'], test_sources=['3'])


def _with_extra_data(dataset):
    """Attach a few extra-data rows per target, tagged by target, as a dataset with extra data does."""
    extra = []
    for target, frame in zip(dataset['target_sources'], dataset['target_data']):
        times = pd.to_datetime(frame[DATES]).iloc[::50]
        extra.append(pd.DataFrame({DATES: times.to_numpy(), 'source': target, 'target': target,
                                   'name': 'inspection', 'type': 'isolated'}))
    dataset['target_extra_data'] = extra
    return dataset


def _raw_targets(dataset):
    """Each target's data as the flavors score it: dates as the index, before preprocessing."""
    raw = {}
    for target, frame in zip(dataset['target_sources'], dataset['target_data']):
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop(DATES)))
        raw[target] = frame
    return raw


def _check_calls(label, dataset, expect_extra=True):
    assert CALLS, f"{label}: transform was never called"
    raw = _raw_targets(dataset)
    n_extra = {target: len(e) for target, e in zip(dataset['target_sources'],
                                                  dataset.get('target_extra_data', []))}
    if not expect_extra:
        for call in CALLS:
            assert call['extra_data'] is None, f"{label}: extra_data should be None"
            assert call['target_data'] is not None and len(call['target_data']) == call['n_scores']
        print(f"OK: {label}: {len(CALLS)} transform call(s) received target_data and "
              f"extra_data=None")
        return
    for call in CALLS:
        target_data, extra = call['target_data'], call['extra_data']
        assert extra is not None, f"{label}: extra_data missing"
        assert isinstance(extra.index, pd.DatetimeIndex) and DATES not in extra.columns, label
        targets = set(extra['target'])
        assert len(targets) == 1, f"{label}: extra data of several targets in one call"
        target = targets.pop()
        assert len(extra) == n_extra[target], f"{label}: {target} got {len(extra)} extra rows"
        assert target_data is not None, f"{label}: target_data missing"
        assert isinstance(target_data.index, pd.DatetimeIndex), label
        assert len(target_data) == call['n_scores'], f"{label}: target_data not one row per score"
        expected = raw[target].loc[target_data.index]
        assert np.allclose(target_data.to_numpy(dtype=float), expected.to_numpy(dtype=float)), \
            f"{label}: target_data is not the raw data of target {target}"
    print(f"OK: {label}: {len(CALLS)} transform call(s) received their own target's raw data "
          f"and extra data")


def _run(label, dataset, method, params, experiment, postprocessor=RecordingPostProcessor,
         **kwargs):
    CALLS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_experiment(dataset=dataset, methods=[method], param_space_dict_per_method=[params],
                       method_names=[method.__name__], experiments=[experiment],
                       experiment_names=[f'TransformAPI_{label}'], MAX_RUNS=2, MAX_JOBS=1,
                       INITIAL_RANDOM=1, postprocessor=postprocessor, **kwargs)


FLAVORS = [
    ('semi_supervised', 'get_semi_dataset', OneClassSVM, {'nu': [0.1]},
     SemiSupervisedPdMExperiment, {}),
    ('supervised', 'get_Classification_dataset', XGBoost, {'n_estimators': [10]},
     SupervisedPdMExperiment, {}),
    ('unsupervised', 'get_unsupervised_dataset', IsolationForestUnsupervised,
     {'n_estimators': [10]}, UnsupervisedPdMExperiment, {}),
    ('auto_profile', 'get_semi_dataset', OneClassSVM, {'nu': [0.1]},
     AutoProfileSemiSupervisedPdMExperiment, {'profile_size': [10, 20]}),
    ('incremental', 'get_semi_dataset', OneClassSVM, {'nu': [0.1]},
     IncrementalSemiSupervisedPdMExperiment,
     {'additional_parameters': {'initial_incremental_window_length': [100],
                                'incremental_window_length': [100],
                                'incremental_slide': [100]}}),
    ('rul', 'get_rul_dataset', XGBoostRUL, {'n_estimators': [10]}, SupervisedRULPdMExperiment, {}),
    ('sa', 'get_SA_dataset', CoxPH, {'alpha': [0.01]}, Supervised_SA_PdMExperiment,
     {'optimization_param': 'IBS', 'thresholder': SurvToRUL}),
]


def test_every_flavor_passes_target_and_extra_data():
    ds = _ims_dataset()
    for label, builder, method, params, experiment, kwargs in FLAVORS:
        dataset, _ = getattr(ds, builder)()
        dataset = _with_extra_data(dataset)
        _run(label, dataset, method, params, experiment, **kwargs)
        _check_calls(label, dataset)


def test_target_data_is_taken_before_preprocessing():
    """With a MinMaxScaler preprocessor the post-processor still gets the unscaled rows."""
    dataset, _ = _ims_dataset().get_semi_dataset()
    dataset = _with_extra_data(dataset)
    _run('semi_minmax', dataset, OneClassSVM, {'nu': [0.1]}, SemiSupervisedPdMExperiment,
         preprocessor=MinMaxScaler)
    _check_calls('semi_supervised with MinMaxScaler', dataset)


def test_without_extra_data_extra_data_is_none():
    dataset, _ = _ims_dataset().get_semi_dataset()
    _run('semi_no_extra', dataset, OneClassSVM, {'nu': [0.1]}, SemiSupervisedPdMExperiment)
    _check_calls('semi_supervised without extra data', dataset, expect_extra=False)


def test_loaded_mlflow_pipeline_accepts_extra_data():
    ds = _ims_dataset()
    dataset, test_dataset = ds.get_semi_dataset()
    dataset = _with_extra_data(dataset)
    test_dataset = _with_extra_data(test_dataset)
    _run('mlflow', dataset, OneClassSVM, {'nu': [0.1]}, SemiSupervisedPdMExperiment,
         postprocessor=ExtraRowCountPostProcessor)

    experiment = mlflow.get_experiment_by_name('TransformAPI_mlflow OneClassSVM')
    run_id = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                order_by=["start_time DESC"], max_results=1).iloc[0].run_id
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/best_pdm_pipeline")

    target = test_dataset['target_data'][0].copy()
    target.index = pd.DatetimeIndex(pd.to_datetime(target.pop(DATES)))
    extra = test_dataset['target_extra_data'][0].set_index(DATES)
    model_input = {'target_data': target, 'source': test_dataset['target_sources'][0],
                   'event_data': test_dataset['event_data']}
    without = np.array(loaded.predict(model_input)['scores'])
    with_extra = np.array(loaded.predict({**model_input, 'extra_data': extra})['scores'])
    assert np.allclose(with_extra - without, len(extra))
    print(f"OK: loaded MLflow pipeline passed {len(extra)} extra rows to the post-processor")


def main():
    tests = [
        test_legacy_signature_still_works,
        test_new_signature_receives_target_and_extra_data,
        test_wrong_number_of_scores_raises,
        test_target_data_is_matched_to_the_scored_rows,
        test_unmatched_target_data_warns_only_when_needed,
        test_merged_rows_give_no_target_data,
        test_extra_data_must_be_indexed_by_time,
        test_every_flavor_passes_target_and_extra_data,
        test_target_data_is_taken_before_preprocessing,
        test_without_extra_data_extra_data_is_none,
        test_loaded_mlflow_pipeline_accepts_extra_data,
    ]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print("All post-processor API tests passed.")


if __name__ == '__main__':
    # Run from the imported module rather than from __main__: MLflow serialises the logged
    # pipeline with cloudpickle, which copies classes defined in __main__ by value, globals
    # included, and the recording post-processor would then append to a copy of CALLS.
    import test_postprocessor_api
    test_postprocessor_api.main()
