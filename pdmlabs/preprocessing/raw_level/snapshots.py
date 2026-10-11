"""Snapshot datasets: references to snapshots and their conversion to feature rows.

A snapshot dataset is a folder per source (one file per snapshot), a list of DataFrames
per source, or one folder whose subfolders are the sources (:func:`discover_sources`).

The Dataset class does not read the snapshots: it builds a placeholder frame with one row
per snapshot (timestamp, source and a :class:`SnapshotRef`), so that episodes, splits,
predictive horizon and slide are derived as for any dataset. The raw-level step of the
pipeline reads the snapshots later and replaces each reference by one row of features
(:func:`featurize`).

Features are cached in memory per snapshot and raw-level configuration, so within a Python
session each snapshot is read once per configuration.
"""

import hashlib
import os
import re
import warnings
from datetime import datetime

import cloudpickle
import pandas as pd

#: Column of the placeholder frame that holds the snapshot references.
SNAPSHOT_COLUMN = '__snapshot__'

#: Keys accepted in ``Dataset(snapshot_options=...)``.
SNAPSHOT_OPTIONS = ('timestamp_format', 'datetime_column', 'interval', 'start',
                    'read_kwargs', 'reader')

#: Start of the artificial timeline when ``interval`` is used without ``start``.
DEFAULT_START = '2000-01-01 00:00:00'

_feature_cache = {}
_reader_wrappers = {}
_loaded_readers = {}


class _ReaderByValue:
    """Pickles a reader with cloudpickle, so that trial workers (MAX_JOBS > 1) also get a
    reader defined in the user's script. Shared by all references to a reader, so the reader
    is pickled once per transfer.
    """

    def __init__(self, reader):
        self.reader = reader

    def __reduce__(self):
        return _load_reader, (cloudpickle.dumps(self.reader),)


def _load_reader(pickled):
    # One object per distinct reader, so that the in-memory cache, which keys on the reader,
    # keeps its entries across the trials a worker runs.
    if pickled not in _loaded_readers:
        _loaded_readers[pickled] = cloudpickle.loads(pickled)
    return _loaded_readers[pickled]


class SnapshotRef:
    """A reference to one snapshot: a file on disk or an in-memory DataFrame.

    ``key`` identifies the snapshot and its content (file path, size, modification time and
    read options, or source, position and content hash), so a changed file never gets its
    features from the cache.
    """

    def __init__(self, key, path=None, frame=None, read_kwargs=None, reader=None, time_column=None):
        self.key = key
        self.path = path
        self.frame = frame
        self.read_kwargs = dict(read_kwargs or {})
        self.reader = reader
        self.time_column = time_column

    def load(self) -> pd.DataFrame:
        """Return the snapshot's samples."""
        if self.frame is not None:
            return self.frame
        reader = self.reader if self.reader is not None else pd.read_csv
        return reader(self.path, **self.read_kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.reader is not None:
            state['reader'] = _reader_wrappers.setdefault(self.reader, _ReaderByValue(self.reader))
        return state

    def __repr__(self):
        return f'SnapshotRef({self.key!r})'


def build_snapshot_frame(data: dict, datetime_column: str, source_column: str,
                         options: dict) -> pd.DataFrame:
    """Build the placeholder frame of a snapshot dataset.

    Args:
        data (dict): Source name -> folder path (one file per snapshot) or list of
            DataFrames (one per snapshot).
        datetime_column (str): Name of the timestamp column of the frame.
        source_column (str): Name of the source column of the frame.
        options (dict): The ``snapshot_options`` of :class:`pdmlabs.utils.dataset.Dataset`.
            Folder files are ordered by time with ``timestamp_format``, otherwise by the
            natural order of their names (``2.csv`` before ``10.csv``). With
            ``datetime_column`` a snapshot's time is the column's first value; with
            ``interval`` snapshot ``i`` is stamped ``start + i * interval`` (``start``
            defaults to ``DEFAULT_START``). ``reader(path, **read_kwargs)`` returns one
            snapshot (``pd.read_csv`` by default).

    Returns:
        pd.DataFrame: One row per snapshot with columns ``datetime_column``,
            ``source_column`` and ``SNAPSHOT_COLUMN``, in source order then time order.

    Raises:
        ValueError: On unknown options, a missing or ambiguous timestamp option, or
            timestamps that are not strictly increasing within a source.
        TypeError: If a source is neither a folder path nor a list of DataFrames.
    """
    unknown = set(options) - set(SNAPSHOT_OPTIONS)
    if unknown:
        raise ValueError(f'Unknown snapshot_options {sorted(unknown)}; expected a subset of '
                         f'{list(SNAPSHOT_OPTIONS)}.')
    timestamp_options = [name for name in ('timestamp_format', 'datetime_column', 'interval')
                         if options.get(name) is not None]
    if len(timestamp_options) != 1:
        raise ValueError(
            'snapshot_options must give exactly one way to timestamp the snapshots: '
            "'timestamp_format' (parse the file names), 'datetime_column' (a column inside "
            "each snapshot) or 'interval' (artificial spacing), got "
            f'{timestamp_options or "none"}.'
        )
    if options.get('start') is not None and options.get('interval') is None:
        raise ValueError("snapshot_options 'start' is only used together with 'interval'.")

    frames = []
    for source, snapshots in data.items():
        if isinstance(snapshots, (str, os.PathLike)):
            refs, timestamps = _folder_snapshots(str(source), os.fspath(snapshots), options)
        elif isinstance(snapshots, (list, tuple)):
            refs, timestamps = _frame_snapshots(str(source), list(snapshots), options)
        else:
            raise TypeError(
                f"Snapshots of source '{source}' must be a folder path or a list of DataFrames, "
                f'got {type(snapshots).__name__}.'
            )
        if len(refs) == 0:
            raise ValueError(f"Source '{source}' has no snapshots.")
        timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))
        if not timestamps.is_monotonic_increasing or not timestamps.is_unique:
            raise ValueError(
                f"The snapshot timestamps of source '{source}' must be strictly increasing, in "
                'the order the snapshots are given.'
            )
        frames.append(pd.DataFrame({
            datetime_column: timestamps,
            source_column: str(source),
            SNAPSHOT_COLUMN: refs,
        }))
    return pd.concat(frames, ignore_index=True)


def discover_sources(root) -> dict:
    """Find the sources of a snapshot dataset stored under a single folder.

    Every folder under ``root`` that directly holds files is a source, named by its relative
    path (e.g. ``'1st_test/1st_test'`` in IMS); a ``root`` that only holds files is one source.
    Hidden files are skipped, and files directly in ``root`` (e.g. a readme) are ignored with
    a warning.

    Args:
        root (str | os.PathLike): The dataset's folder.

    Returns:
        dict: Source name -> folder, in natural order of the names.

    Raises:
        ValueError: If ``root`` is a file, does not exist or holds no files.
    """
    root = os.fspath(root)
    if os.path.isfile(root):
        raise ValueError(f"'{root}' is a file. Pass a DataFrame, a dict of sources (folders or lists of "
                         'DataFrames), or a folder that holds the snapshot files of every source.')
    if not os.path.isdir(root):
        raise ValueError(f"Snapshot dataset folder '{root}' does not exist.")

    sources, root_files = {}, []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if not name.startswith('.'))
        files = [name for name in files if not name.startswith('.')]
        if not files:
            continue
        if os.path.normpath(directory) == os.path.normpath(root):
            root_files = files
        else:
            sources[os.path.relpath(directory, root).replace(os.sep, '/')] = directory
    if not sources:
        if not root_files:
            raise ValueError(f"No snapshot files found under '{root}'.")
        return {os.path.basename(os.path.normpath(root)): root}
    if root_files:
        warnings.warn(f"Ignored {len(root_files)} file(s) directly in '{root}' (e.g. {sorted(root_files)[:3]}): "
                      'the snapshot files are in its subfolders, one source per subfolder.')
    return dict(sorted(sources.items(), key=lambda item: _natural_key(item[0])))


def _folder_snapshots(source, folder, options):
    if os.path.isfile(folder):
        raise ValueError(f"The path of source '{source}' is a file ('{folder}'); expected a folder holding "
                         'one file per snapshot.')
    if not os.path.isdir(folder):
        raise ValueError(f"Snapshot folder '{folder}' of source '{source}' does not exist.")
    names = [name for name in os.listdir(folder)
             if not name.startswith('.') and os.path.isfile(os.path.join(folder, name))]
    timestamp_format = options.get('timestamp_format')
    if timestamp_format is not None:
        parsed = sorted((_parse_file_name(name, timestamp_format), name) for name in names)
        names = [name for _, name in parsed]
    else:
        names = sorted(names, key=_natural_key)

    refs = []
    read_key = _read_key(options)
    for name in names:
        path = os.path.abspath(os.path.join(folder, name))
        stat = os.stat(path)
        refs.append(SnapshotRef(
            key=f'file:{path}:{stat.st_size}:{stat.st_mtime_ns}:{read_key}',
            path=path,
            read_kwargs=options.get('read_kwargs'),
            reader=options.get('reader'),
            time_column=options.get('datetime_column'),
        ))

    if timestamp_format is not None:
        timestamps = [timestamp for timestamp, _ in parsed]
    elif options.get('datetime_column') is not None:
        timestamps = [_first_timestamp(ref.load(), options['datetime_column'], source)
                      for ref in refs]
    else:
        timestamps = _interval_timestamps(len(refs), options)
    return refs, timestamps


def _frame_snapshots(source, snapshots, options):
    if not all(isinstance(snapshot, pd.DataFrame) for snapshot in snapshots):
        raise TypeError(f"Every snapshot of source '{source}' must be a pandas DataFrame.")
    if options.get('timestamp_format') is not None:
        raise ValueError(
            "snapshot_options 'timestamp_format' parses file names and only applies to folder "
            f"sources; source '{source}' is a list of DataFrames. Use 'datetime_column' or "
            "'interval' instead."
        )

    refs = []
    time_column = options.get('datetime_column')
    for index, snapshot in enumerate(snapshots):
        content = hashlib.sha1(pd.util.hash_pandas_object(snapshot, index=True).to_numpy().tobytes())
        content.update(repr(list(snapshot.columns)).encode())
        refs.append(SnapshotRef(
            key=f'frame:{source}#{index}:{content.hexdigest()}:{time_column!r}',
            frame=snapshot,
            time_column=time_column,
        ))

    if options.get('datetime_column') is not None:
        timestamps = [_first_timestamp(snapshot, options['datetime_column'], source)
                      for snapshot in snapshots]
    else:
        timestamps = _interval_timestamps(len(refs), options)
    return refs, timestamps


def _read_key(options):
    """Describe how snapshot files are read (the reader's name, ``read_kwargs`` and the time
    column that is removed), so that a file read differently never shares a cached result."""
    reader = options.get('reader')
    name = 'pandas.read_csv' if reader is None else \
        f"{getattr(reader, '__module__', '')}.{getattr(reader, '__qualname__', type(reader).__qualname__)}"
    read_kwargs = sorted((options.get('read_kwargs') or {}).items())
    return f'{name}:{read_kwargs!r}:{options.get("datetime_column")!r}'


def _parse_file_name(name, timestamp_format):
    candidates = [name, os.path.splitext(name)[0]]
    for candidate in candidates:
        try:
            return datetime.strptime(candidate, timestamp_format)
        except ValueError:
            pass
    raise ValueError(f"Snapshot file name '{name}' does not match timestamp_format "
                     f"'{timestamp_format}'.")


def _natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def _first_timestamp(snapshot, column, source):
    if column not in snapshot.columns:
        raise ValueError(f"A snapshot of source '{source}' has no '{column}' column "
                         "(snapshot_options['datetime_column']).")
    if pd.api.types.is_numeric_dtype(snapshot[column]):
        raise ValueError(
            f"The '{column}' column of the snapshots of source '{source}' holds numbers, whose "
            "unit (seconds, milliseconds, ...) is ambiguous. Convert it to datetimes first, e.g. "
            "with pd.to_datetime(values, unit='s') in snapshot_options['reader'] or before "
            'passing the DataFrames.'
        )
    return pd.to_datetime(snapshot[column].iloc[0])


def _interval_timestamps(count, options):
    start = pd.Timestamp(options.get('start') or DEFAULT_START)
    return [start + index * pd.Timedelta(options['interval']) for index in range(count)]


def featurize(frame: pd.DataFrame, raw_step) -> pd.DataFrame:
    """Replace the snapshot references of a frame by one row of features each.

    Rows are matched by position, so the rows, order and index of ``frame`` are kept, and
    the feature columns follow its other columns. A frame without snapshot references is
    returned unchanged.

    Args:
        frame (pd.DataFrame): Rows of a snapshot dataset.
        raw_step (RawLevelPreProcessorInterface): The feature extractor.

    Returns:
        pd.DataFrame: The featurized frame.
    """
    if SNAPSHOT_COLUMN not in frame.columns:
        return frame

    token = _cache_token(raw_step)
    refs = list(frame[SNAPSHOT_COLUMN])
    rows = [_features_of(ref, raw_step, token) for ref in refs]
    columns = list(rows[0]) if rows else []
    if rows and not columns:
        raise ValueError(
            f'{raw_step} returned no features for snapshot {refs[0]!r}, e.g. because it has no '
            "numeric channels. For snapshot files, set how they are read with snapshot_options["
            "'read_kwargs'], e.g. {'sep': '\\t', 'header': None} for tab-separated files without a header."
        )
    for ref, row in zip(refs, rows):
        if list(row) != columns:
            only_first = [name for name in columns if name not in row]
            only_this = [name for name in row if name not in columns]
            raise ValueError(
                f'{raw_step} returned {len(columns)} features for snapshot {refs[0]!r} but '
                f'{len(row)} for snapshot {ref!r} (only in the first: {only_first[:5]}, only in the '
                f'second: {only_this[:5]}). Every snapshot must have the same channels; select '
                "them, e.g. with read_kwargs={'usecols': [...]} or snapshot_options['reader']."
            )

    rest = frame.drop(columns=[SNAPSHOT_COLUMN])
    clashes = set(rest.columns) & set(columns)
    if clashes:
        raise ValueError(f'Feature columns {sorted(map(str, clashes))} clash with existing '
                         'columns of the data.')
    features = pd.DataFrame(rows, columns=columns)
    result = pd.concat([rest.reset_index(drop=True), features], axis=1)
    result.index = frame.index
    return result


def _cache_token(raw_step):
    """Identify a raw-level configuration by its class name and get_params()."""
    cls = type(raw_step)
    return f'{cls.__module__}.{cls.__qualname__}{sorted(raw_step.get_params().items())!r}'


def _features_of(ref, raw_step, token):
    # The class and the reader also count as objects: one defined again under the same name
    # (a notebook cell run again, a reader made by a factory) gets its own entries.
    cache_key = (ref.key, ref.reader, type(raw_step), token)
    if cache_key in _feature_cache:
        return _feature_cache[cache_key]

    snapshot = ref.load()
    if ref.time_column is not None:
        # The timestamp is not a measurement; as a feature it would tell the method how far
        # into the run a snapshot was recorded.
        snapshot = snapshot.drop(columns=[ref.time_column])
    features = raw_step.extract(snapshot)
    if not isinstance(features, dict):
        raise TypeError(f'{raw_step}.extract() must return a dict, got {type(features).__name__}.')
    _feature_cache[cache_key] = features
    return features
