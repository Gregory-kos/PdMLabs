"""Normalise what a method or post-processor returns into the list the flavors and evaluators expect."""
import numpy as np
import pandas as pd


def as_score_list(scores, n_rows=None, where='predict'):
    """Return `scores` as a plain list, one entry per input row.

    1-D numeric output (list, tuple, numpy array, pandas Series, or an (n, 1) column) becomes a
    list of Python floats. Output with per-row structure (e.g. survival curves shaped (n, 2, T))
    is only split into a list of rows and left otherwise untouched.
    """
    if isinstance(scores, (pd.Series, pd.DataFrame)):
        scores = scores.to_numpy()
    if isinstance(scores, np.ndarray):
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        scores = scores.tolist() if scores.ndim == 1 else list(scores)
    elif not isinstance(scores, list):
        scores = list(scores)
    if n_rows is not None and len(scores) != n_rows:
        raise ValueError(f'{where} returned {len(scores)} scores for {n_rows} input rows')
    return scores
