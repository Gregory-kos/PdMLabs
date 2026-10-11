"""Chaining of the raw-level step in front of the record-level preprocessor.

On a snapshot dataset the pipeline's preprocessor is a :class:`RawThenRecordPreProcessor`:
it turns the snapshot references into feature rows (raw-level step), then hands them to
the record-level preprocessor, so no experiment has to change.

An ``additional_parameters`` key ``preprocessor_raw_<param>`` reaches the raw-level step
as ``<param>`` and is logged to MLflow under that key; all other preprocessor parameters go
to the record-level step, which therefore must not have parameters starting with ``raw_``.
"""

import pandas as pd

from pdmlabs.preprocessing.record_level.record_level_pre_processor import RecordLevelPreProcessorInterface
from pdmlabs.preprocessing.raw_level.snapshots import SNAPSHOT_COLUMN, featurize
from pdmlabs.pdm_evaluation_types.types import EventPreferences

#: Prefix that routes a preprocessor parameter to the raw-level step.
RAW_PARAM_PREFIX = 'raw_'


class RawThenRecordPreProcessor(RecordLevelPreProcessorInterface):
    """Featurize snapshots (raw-level step), then apply a record-level preprocessor.

    The raw-level step is stateless, so fit() only fits the record-level step, on the
    featurized frames it is given -- the same training frames as without snapshots.

    Attributes:
        raw (pdmlabs.preprocessing.raw_level.raw_level_pre_processor.RawLevelPreProcessorInterface):
            The raw-level step.
        record (RecordLevelPreProcessorInterface): The record-level step.
    """

    def __init__(self, event_preferences: EventPreferences, raw_cls, record_cls, **params):
        """Build both steps from the routed parameters.

        Args:
            event_preferences (EventPreferences): Event configuration dict.
            raw_cls (type): Raw-level preprocessor class.
            record_cls (type): Record-level preprocessor class.
            **params: ``raw_<param>`` for the raw-level step, the rest for the
                record-level step.
        """
        super().__init__(event_preferences=event_preferences)
        raw_params = {key[len(RAW_PARAM_PREFIX):]: value for key, value in params.items()
                      if key.startswith(RAW_PARAM_PREFIX)}
        record_params = {key: value for key, value in params.items()
                         if not key.startswith(RAW_PARAM_PREFIX)}
        self.raw = raw_cls(**raw_params)
        self.record = record_cls(event_preferences=event_preferences, **record_params)
        # Raw-level features of the data fit() was given, to give a clear error when other
        # data yields different ones (e.g. a source with a different number of channels).
        self.fitted_features = None

    def fit(self, historic_data: list, historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges=None) -> None:
        """Featurize the historic frames and fit the record-level step on them.

        Args:
            historic_data (list[pd.DataFrame]): Training frames, one per source.
            historic_sources (list[str]): Source identifiers.
            event_data (pd.DataFrame): Event log.
            anomaly_ranges (optional): Passed on to the record-level step when given.
        """
        featurized = [featurize(frame, self.raw) for frame in historic_data]
        self.fitted_features = None
        for frame, features in zip(historic_data, featurized):
            self._check_features(_added_columns(frame.columns, features.columns), fitting=True)
        if anomaly_ranges is None:
            self.record.fit(featurized, historic_sources, event_data)
        else:
            self.record.fit(featurized, historic_sources, event_data, anomaly_ranges)

    def transform(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame) -> pd.DataFrame:
        """Featurize the frame, then apply the record-level step.

        Args:
            target_data (pd.DataFrame): Rows of a snapshot dataset (or already featurized rows).
            source (str): Source identifier.
            event_data (pd.DataFrame): Event log.

        Returns:
            pd.DataFrame: One row of features per input row, same index.
        """
        featurized = featurize(target_data, self.raw)
        self._check_features(_added_columns(target_data.columns, featurized.columns))
        return self.record.transform(featurized, source, event_data)

    def transform_one(self, new_sample: pd.Series, source: str, is_event: bool) -> pd.Series:
        """Featurize a single row, then apply the record-level step to it.

        Args:
            new_sample (pd.Series): One row of a snapshot dataset.
            source (str): Source identifier.
            is_event (bool): Whether this row is an event.

        Returns:
            pd.Series: The transformed row.
        """
        if SNAPSHOT_COLUMN in new_sample.index:
            featurized = featurize(new_sample.to_frame().T, self.raw).iloc[0]
            self._check_features(_added_columns(new_sample.index, featurized.index))
            new_sample = featurized
        return self.record.transform_one(new_sample, source, is_event)

    def _check_features(self, features, fitting=False):
        """Record (when fitting) or compare the raw-level features of some data, and raise a
        ValueError naming the differing features if they differ from the fitted ones.

        ``features`` is None when the data held no snapshot references.
        """
        if features is None:
            return
        if fitting and self.fitted_features is None:
            self.fitted_features = features
            return
        if self.fitted_features is None or features == self.fitted_features:
            return
        only_fitted = [name for name in self.fitted_features if name not in features]
        only_this = [name for name in features if name not in self.fitted_features]
        detail = (f'only in the fitted data: {only_fitted[:5]}; only in this data: {only_this[:5]}'
                  if only_fitted or only_this else 'the same features in another order')
        raise ValueError(
            f'{self.raw} gives {len(features)} features for this data, but the pipeline was fitted on '
            f'data with {len(self.fitted_features)} features ({detail}). The sources do not have the '
            "same channels: select the same channels for every source, e.g. with "
            "read_kwargs={'usecols': [...]} or snapshot_options['reader']."
        )

    def get_params(self) -> dict:
        """Return the parameters of both steps (raw-level ones prefixed with ``raw_``).

        Returns:
            dict: Merged hyperparameters.
        """
        params = {f'{RAW_PARAM_PREFIX}{key}': value for key, value in self.raw.get_params().items()}
        params.update(self.record.get_params())
        return params

    def __str__(self) -> str:
        """Return the names of both steps.

        Returns:
            str: '<raw>+<record>', e.g. 'Statistical+MinMaxScaler'.
        """
        return f'{self.raw}+{self.record}'


class RawThenRecordFactory:
    """Build :class:`RawThenRecordPreProcessor` objects the way experiments build preprocessors.

    Experiments create their preprocessor as
    ``pipeline.preprocessor(event_preferences=..., **preprocessor_params)``; this
    picklable callable takes that place in the pipeline.
    """

    def __init__(self, raw_cls, record_cls):
        self.raw_cls = raw_cls
        self.record_cls = record_cls

    def __call__(self, event_preferences: EventPreferences, **params) -> RawThenRecordPreProcessor:
        return RawThenRecordPreProcessor(event_preferences, self.raw_cls, self.record_cls, **params)

    def __repr__(self):
        return f'RawThenRecordFactory({self.raw_cls.__name__}, {self.record_cls.__name__})'


def _added_columns(before, after):
    """Return the feature columns featurize() added, or None if there were no snapshots."""
    if SNAPSHOT_COLUMN not in before:
        return None
    return [name for name in after if name not in before]


def featurize_if_snapshots(frame: pd.DataFrame, preprocessor) -> pd.DataFrame:
    """Featurize a frame with the preprocessor's raw-level step, if it has one.

    Only the stateless raw-level step runs; nothing is fitted.

    Args:
        frame (pd.DataFrame): Rows of a (possibly snapshot) dataset.
        preprocessor: The pipeline's preprocessor object.

    Returns:
        pd.DataFrame: The featurized frame, or ``frame`` unchanged.
    """
    if isinstance(preprocessor, RawThenRecordPreProcessor):
        return featurize(frame, preprocessor.raw)
    return frame
