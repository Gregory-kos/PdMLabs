"""Interface definition for raw-level (snapshot) preprocessing components.

A raw-level preprocessor turns one *snapshot* (e.g. one IMS file of 20 480 samples per
channel) into one row of features. It is the first step of a pipeline on a snapshot
dataset, before the record-level preprocessor.

The features of a snapshot must depend on that snapshot alone: the step runs on train,
validation and test snapshots alike, so anything learned across snapshots would leak
between the splits. Learned transformations belong in the record-level step.
"""

import abc

import pandas as pd


class RawLevelPreProcessorInterface(abc.ABC):
    """Abstract base class for raw-level (snapshot to feature row) preprocessing.

    Subclasses must implement:
    - extract(): Compute the features of one snapshot
    - get_params(): Return hyperparameters as dict
    - __str__(): Return a short name

    Hyperparameters are passed as keyword arguments to the constructor. When a
    pipeline is tuned, they come from ``additional_parameters`` keys of the form
    ``preprocessor_raw_<param>`` (e.g. ``preprocessor_raw_n_bins: [10, 20]``),
    which is also the name they are logged under in MLflow.
    Parameter names must not contain ``method``, ``postprocessor`` or ``thresholder``,
    because pipeline parameters are routed to their step by substring.

    Examples:
        >>> class MeanPerChannel(RawLevelPreProcessorInterface):
        ...     def extract(self, snapshot):
        ...         channels = self.numeric_channels(snapshot)
        ...         return {f'mean_{c}': float(channels[c].mean()) for c in channels.columns}
        ...     def get_params(self):
        ...         return {}
        ...     def __str__(self):
        ...         return 'MeanPerChannel'
    """

    @abc.abstractmethod
    def extract(self, snapshot: pd.DataFrame) -> dict:
        """Compute the features of a single snapshot.

        Args:
            snapshot (pd.DataFrame): The samples of one snapshot, one column per
                channel. The ``datetime_column`` of the snapshot options is removed
                beforehand; other non-numeric columns are ignored.

        Returns:
            dict: Feature name -> float. Every snapshot of a dataset must yield the
                same feature names in the same order.
        """
        pass

    @abc.abstractmethod
    def get_params(self) -> dict:
        """Return hyperparameters as a dictionary.

        Used for logging to MLflow and to key the feature cache, so it must contain
        every parameter that changes the output of extract().

        Returns:
            dict: Hyperparameters, e.g. {'n_bins': 20}.
        """
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        """Return a short name/identifier for this preprocessor.

        Returns:
            str: Preprocessor name.
        """
        pass

    @staticmethod
    def numeric_channels(snapshot: pd.DataFrame) -> pd.DataFrame:
        """Return the numeric columns (channels) of a snapshot."""
        return snapshot.select_dtypes(include='number')
