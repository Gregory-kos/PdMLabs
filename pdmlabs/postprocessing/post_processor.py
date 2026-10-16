"""Abstract base class interface for post-processors.

Post-processors transform anomaly detection scores to improve decision quality.
They operate on MODEL OUTPUT (anomaly scores), not raw sensor data:

- Smoothing: Reduce score variance (moving average, self-tuning normalization)
- Thresholding: Convert scores to binary anomaly labels or adaptive thresholds
- Normalization: Scale scores to [0, 1] for consistent interpretation
- Filtering: Remove noise or refine predictions across time windows

Typical pipeline: RAW DATA -> (Preprocessing) -> SENSOR FEATURES -> (Detection Model) -> 
ANOMALY SCORES -> (PostProcessing) -> FINAL ANOMALY LABELS/CONFIDENCE
"""

import abc
import inspect
import warnings

import pandas as pd

from pdmlabs.pdm_evaluation_types.types import EventPreferences
from pdmlabs.utils.scores import as_score_list


class PostProcessorInterface(abc.ABC):
    """Abstract base class for anomaly score post-processors.
    
    Post-processors operate on model outputs (anomaly scores) to improve:
    - Score quality (smoothing, normalization)
    - Interpretability (thresholding to binary labels)
    - Robustness (adaptive thresholds based on history)
    
    Each post-processor must implement fit/transform in two modes:
    - Batch mode: transform() processes many scores at once
    - Online/streaming mode: transform_one() processes one score at a time
    
    Attributes:
        event_preferences (EventPreferences): Event configuration dict
        needs_target_data (bool): Set to True by a post-processor whose `transform` relies on
            `target_data`, so that `postprocess_scores` warns when it has to pass None. False
            (the default) keeps that warning quiet for post-processors that ignore it.
    """
    needs_target_data = False

    def __init__(self, event_preferences: EventPreferences):
        """Initialize post-processor.
        
        Args:
            event_preferences (EventPreferences): Event configuration dict.
        """
        self.event_preferences = event_preferences

    @abc.abstractmethod
    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame, anomaly_ranges=None) -> None:
        """Fit post-processor on training data (anomaly scores or raw data).
        
        Some post-processors are stateless and fit() does nothing. Others compute
        statistics from train data to calibrate thresholds or normalization.
        
        Args:
            historic_data (list[pd.DataFrame]): Training data, one per source.
            historic_sources (list[str]): Source identifiers.
            event_data (pd.DataFrame): Event log with failure/reset events.
            anomaly_ranges: Optional data structure marking normal/anomalous regions.
        """
        pass

    @abc.abstractmethod
    def transform(self, scores: list[float], source: str, event_data: pd.DataFrame,
                  target_data: pd.DataFrame = None, extra_data: pd.DataFrame = None,
                  target_source: str = None) -> list[float]:
        """Transform batch of anomaly scores (offline/batch mode).

        Args:
            scores (list[float]): Anomaly scores to post-process.
            source (str): Source identifier (e.g., 'bearing_1'). In the semi-supervised and
                supervised flavors this is the source the model was fitted on (e.g. 'train'),
                not the name of the target being scored.
            event_data (pd.DataFrame): Event log (unused by most post-processors).
            target_data (pd.DataFrame, optional): The target's data the scores were computed
                from, as given to the pipeline, before any preprocessing: one row per score, in
                the same order, indexed by time. None when the caller has none to give.
            extra_data (pd.DataFrame, optional): The extra data passed to the Dataset
                (``extra_data``) for this same target, indexed by time, or None when no extra
                data was given. It holds every row of the target, regardless of time: a
                post-processor that must not use future information has to restrict itself to
                the rows up to each score's timestamp.
            target_source (str, optional): Name of the target being scored (an entry of the
                dataset's ``target_sources``), which `source` is not in every flavor; e.g. to
                find this target's own events in `event_data`. None when the caller does not
                know it.

        Returns:
            list[float]: Post-processed scores (same length as input).
        """
        pass


    @abc.abstractmethod
    def transform_one(self, score_point: float, source: str, is_event: bool) -> float:
        """Transform single anomaly score (online/streaming mode).
        
        Used when processing one score at a time (e.g., real-time anomaly detection).
        Maintains internal buffer for context-aware transformations (moving average, etc).
        
        Args:
            score_point (float): Single anomaly score to post-process.
            source (str): Source identifier (used to maintain per-source state).
            is_event (bool): Whether this score is from an event sample.
        
        Returns:
            float: Post-processed score.
        """
        pass


    @abc.abstractmethod
    def get_params(self):
        """Return hyperparameters.
        
        Returns:
            dict: Hyperparameter names and values (e.g., {'window_length': 5}).
        """
        pass
    

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return post-processor name.

        Returns:
            str: Human-readable name (e.g., 'Moving_Average').
        """
        pass


#: Arguments of `transform` added after its original three, passed only to a `transform` that
#: accepts them.
CONTEXT_ARGUMENTS = ('target_data', 'extra_data', 'target_source')


def _accepted_context_arguments(transform):
    """Return which of ``target_data``/``extra_data``/``target_source`` the bound `transform`
    accepts."""
    parameters = inspect.signature(transform).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return set(CONTEXT_ARGUMENTS)
    # They are passed by keyword, so a positional-only parameter cannot receive them.
    by_keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {name for name in CONTEXT_ARGUMENTS
            if name in parameters and parameters[name].kind in by_keyword}


def _rows_behind_scores(target_data, n_scores, scored_index):
    """Return the rows of `target_data` the scores were computed from, one per score, or None.

    A preprocessor that drops rows keeps the timestamps of the rows it keeps, so when
    `target_data` is indexed by time and `scored_index` (the preprocessed index) is a subset of
    that index the rows are matched through it. Otherwise `target_data` is used as is if it has
    exactly one row per score. When neither holds -- e.g. a preprocessor such as `MeanAggregator`
    merged several rows into each new, resampled row -- no row-per-score correspondence exists
    and None is returned. A resampling preprocessor whose bin labels are themselves timestamps of
    `target_data` cannot be told apart from one that dropped rows: each score is then matched to
    the first row of its bin.
    """
    if target_data is None:
        return None
    if scored_index is not None and len(scored_index) == n_scores \
            and isinstance(target_data.index, pd.DatetimeIndex) and target_data.index.is_unique \
            and pd.Index(scored_index).isin(target_data.index).all():
        return target_data.loc[scored_index]
    if len(target_data) == n_scores:
        return target_data
    return None


def postprocess_scores(postprocessor, scores, source, event_data, target_data=None,
                       extra_data=None, scored_index=None, target_source=None):
    """Run ``postprocessor.transform`` and return its output as a list of one score per input.

    `target_data`, `extra_data` and `target_source` are only passed to a `transform` that
    accepts them, so a post-processor written for the older
    ``transform(scores, source, event_data)`` signature keeps working unchanged. A `transform`
    accepts them when it names them or takes ``**kwargs``, so a decorator around an older
    `transform` should use `functools.wraps`, which keeps the original signature visible.

    Args:
        postprocessor (PostProcessorInterface): The post-processor to run.
        scores (list[float]): Anomaly scores to post-process.
        source (str): Source identifier passed through to `transform`.
        event_data (pd.DataFrame): Event log passed through to `transform`.
        target_data (pd.DataFrame, optional): The target's data before preprocessing, indexed
            by time. It is narrowed to the rows the scores were computed from (see
            `scored_index`) before reaching `transform`, and passed as None when those rows
            cannot be identified -- with a warning if the post-processor sets
            ``needs_target_data``.
        extra_data (pd.DataFrame, optional): See `PostProcessorInterface.transform`.
        scored_index (pd.Index, optional): Index of the rows that were scored, i.e. of the
            preprocessed data. Needed when preprocessing dropped rows.
        target_source (str, optional): See `PostProcessorInterface.transform`.

    Returns:
        list: The post-processed scores.

    Raises:
        ValueError: When `transform` does not return one score per input score.
    """
    accepted = _accepted_context_arguments(postprocessor.transform)
    context = {}
    if 'target_data' in accepted:
        context['target_data'] = _rows_behind_scores(target_data, len(scores), scored_index)
        if target_data is not None and context['target_data'] is None \
                and getattr(postprocessor, 'needs_target_data', False):
            warnings.warn(
                f"{postprocessor}.transform receives target_data=None: target_data has "
                f"{len(target_data)} row(s) for {len(scores)} score(s) and the scored rows could "
                "not be matched to them by index (e.g. a preprocessor merged rows into new, "
                "resampled ones).",
                stacklevel=2,
            )
    if 'extra_data' in accepted:
        context['extra_data'] = extra_data
    if 'target_source' in accepted:
        context['target_source'] = target_source
    processed = postprocessor.transform(scores, source, event_data, **context)
    return as_score_list(processed, len(scores), f'{postprocessor}.transform')