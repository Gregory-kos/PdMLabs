#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver for the CNC, FEMTO and SCANIA run-to-failure data sets in ./data.

Runs KNN / PB / LOF / IF / OCSVM through the auto-profile semi-supervised and
unsupervised flavors, over every HPO backend in
``pdmlabs.optimization.OPTIMIZER_REGISTRY`` and every seed in ``SEEDS``, then
re-runs each search's winning configuration on the held-out test episodes.

    python run_optimizer_study.py cnc --dry-run
    python run_optimizer_study.py femto --optimizers smac optuna --methods IF LOF
    python run_optimizer_study.py scania --optimizers optuna --methods KNN PB
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
import traceback

import pandas as pd

# pdmlabs imports pyplot without pinning a backend; spawned workers inherit this.
os.environ.setdefault('MPLBACKEND', 'Agg')

from pdmlabs.RunExperiment import run_experiment
from pdmlabs.optimization import OPTIMIZER_REGISTRY
from pdmlabs.utils.dataset import Dataset

from pdmlabs.experiment.batch.auto_profile_semi_supervised_experiment import (
    AutoProfileSemiSupervisedPdMExperiment,
)
from pdmlabs.experiment.batch.unsupervised_experiment import UnsupervisedPdMExperiment

from pdmlabs.method.dist_k_Semi import Distance_Based_Semi
from pdmlabs.method.profile_based import ProfileBased
from pdmlabs.method.lof_semi import LocalOutlierFactor
from pdmlabs.method.isolation_forest import IsolationForest
from pdmlabs.method.ocsvm import OneClassSVM
from pdmlabs.method.isolation_forest_uns import IsolationForestUnsupervised
from pdmlabs.method.lof_uns import LocalOutlierFactorUnsupervised
from pdmlabs.method.dist_k_uns import Distance_Based_Uns

from pdmlabs.utils.automatic_parameter_generation import (
    profile_values,
    semi_technique,
    unsupervised_technique,
)


# --------------------------------------------------------------------------- #
# Fixed settings -- deliberately not exposed on the command line
# --------------------------------------------------------------------------- #
OBJECTIVE = 'AD1_AUC'
MLFLOW_PORT = 8080

# The whole grid is repeated once per seed. `random_state` is what the HPO
# backend seeds its own proposals with, so a fixed one replays the identical
# search on every repeat and the spread between repeats would be hardware noise
# rather than the optimizer's run-to-run variance.
SEEDS = [42, 1337, 2024]

# The train/val/test split is deliberately *not* reseeded per repeat: varying it
# too would fold split variance into the numbers the study is trying to compare.
SPLIT_RANDOM_STATE = 42

# Best-effort ceiling on the number of configurations a single cell can express,
# profile axes included. `cell_space` thins the generated ladders down to it.
TARGET_SPACE_SIZE = 1000

# mango and gpyopt demand INITIAL_RANDOM + MAX_JOBS <= MAX_RUNS, so neither can
# execute the single pinned configuration of the test evaluation.
TEST_OPTIMIZER = 'optuna'

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# predictive_horizon / lead / slide are the values the previous loaders used.
# For cnc and femto every CSV is one whole run-to-failure episode, so no event
# definition is needed: Dataset treats each source as a single failing episode
# by default. scania is stacked and carries its own events -- see
# build_scania_dataset.
DATASETS = {
    'cnc': {
        'folder': os.path.join(DATA_ROOT, 'cnc'),
        'datetime_column': 'timestamp',
        'predictive_horizon': '3 seconds',
        'lead': '1 seconds',
        'slide': 33,
    },
    'femto': {
        'folder': os.path.join(DATA_ROOT, 'femto', 'scenarios'),
        'datetime_column': 'Artificial_timestamp',
        'predictive_horizon': '52 days',
        'lead': '2 days',
        'slide': 117,
    },
    # Built by data/scania/convert_scania.py; 1 time_step = 1 day. The previous
    # loaders' slide of 225 was copied from ims and exceeds every episode here
    # (20-235 readouts), so this is the median number of readouts inside the
    # 48-day horizon instead. It only sizes the VUS buffer, not the objective.
    'scania': {
        'folder': os.path.join(DATA_ROOT, 'scania'),
        'datetime_column': 'timestamp',
        'predictive_horizon': '48 days',
        'lead': '24 hours',
        'slide': 13,
    },
}

# CNC records a millisecond offset from the start of the run rather than a date.
CNC_EPOCH = pd.Timestamp('2025-12-01')


# --------------------------------------------------------------------------- #
# Method registry
# --------------------------------------------------------------------------- #
# The names matter: run_experiment switches the unsupervised constraint function
# on the literal 'KNN' to enforce window > k, and they are what MLflow records.
FLAVORS = {
    'auto_profile': {
        'experiment': AutoProfileSemiSupervisedPdMExperiment,
        'dataset_getter': 'get_semi_dataset',
        'methods': {
            'KNN':   (Distance_Based_Semi, lambda mp: semi_technique('KNN', mp)),
            'PB':    (ProfileBased,        lambda mp: semi_technique('PB', mp)),
            'LOF':   (LocalOutlierFactor,  lambda mp: semi_technique('LOF', mp)),
            'IF':    (IsolationForest,     lambda mp: semi_technique('IF', mp)),
            'OCSVM': (OneClassSVM,         lambda mp: semi_technique('OCSVM', mp)),
        },
    },
    'unsupervised': {
        'experiment': UnsupervisedPdMExperiment,
        'dataset_getter': 'get_unsupervised_dataset',
        # PB and OCSVM are semi-supervised only: the library implements no
        # UnsupervisedMethodInterface for either, and the flavor rejects anything
        # that does not implement it.
        'methods': {
            'KNN': (Distance_Based_Uns,             lambda mp: unsupervised_technique('KNN', mp)),
            'LOF': (LocalOutlierFactorUnsupervised, lambda mp: unsupervised_technique('LOF', mp)),
            'IF':  (IsolationForestUnsupervised,    lambda mp: unsupervised_technique('IF', mp)),
        },
    },
}

ALL_METHODS = ['KNN', 'PB', 'LOF', 'IF', 'OCSVM']
ALL_OPTIMIZERS = sorted(OPTIMIZER_REGISTRY)


# --------------------------------------------------------------------------- #
# Search-space sizing
# --------------------------------------------------------------------------- #
def space_size(space: dict) -> int:
    """Number of configurations a search space can express."""
    size = 1
    for values in space.values():
        size *= max(1, len(values))
    return size


def _thin(values: list, count: int) -> list:
    """``count`` evenly spaced entries of *values*, both endpoints kept.

    The generated ladders are ordered and meaningful -- profile lengths, window
    lengths, neighbour counts -- so thinning has to keep them spanning the same
    range at a coarser step, not truncate them to their head.
    """
    if count >= len(values):
        return list(values)
    if count <= 1:
        return [values[-1]]
    step = (len(values) - 1) / (count - 1)
    return [values[round(index * step)] for index in range(count)]


def shrink_space(space: dict, target: int = TARGET_SPACE_SIZE) -> dict:
    """Thin the longest axis, one value at a time, until the space fits *target*.

    Always taking from the longest axis levels the axes out rather than gutting
    one of them, so a cell spends its budget on every dimension it has. No axis
    goes below two values -- an axis of one is a constant, and the search would
    stop exploring that dimension altogether -- which is why the target is
    best-effort: a space whose axes are all binary can still exceed it.
    """
    space = {key: list(values) for key, values in space.items()}
    while space_size(space) > target:
        key = max(space, key=lambda candidate: len(space[candidate]))
        if len(space[key]) <= 2:
            break
        space[key] = _thin(space[key], len(space[key]) - 1)
    return space


def cell_space(flavor: str, method_name: str, maximum_profile: int, profile_ladder: list):
    """The thinned space of one cell, split the way run_experiment takes it.

    The two profile axes are injected by run_experiment itself, for the
    auto-profile flavor only, so they are thinned together with the method axes
    and handed back as their own arguments.
    """
    space = dict(FLAVORS[flavor]['methods'][method_name][1](maximum_profile))
    method_keys = list(space)
    if flavor == 'auto_profile':
        space['profile_size'] = list(profile_ladder)
        space['initial_profile_size'] = list(profile_ladder)

    space = shrink_space(space)
    method_space = {key: space[key] for key in method_keys}
    return method_space, space.get('profile_size'), space.get('initial_profile_size')


# --------------------------------------------------------------------------- #
# Data set
# --------------------------------------------------------------------------- #
def build_dataset(name: str) -> Dataset:
    """One Dataset over every CSV of the data set folder, one source per file."""
    if name == 'scania':
        return build_scania_dataset()

    spec = DATASETS[name]
    frames = []
    for file_name in sorted(os.listdir(spec['folder'])):
        if not file_name.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(spec['folder'], file_name)).dropna()
        if name == 'cnc':
            df['timestamp'] = CNC_EPOCH + pd.to_timedelta(df['timestamp'], unit='ms')
        df['source'] = file_name.split('.')[0]
        frames.append(df)

    if not frames:
        raise SystemExit(f"no CSV files under {spec['folder']}")

    return Dataset(
        data=pd.concat(frames, ignore_index=True),
        datetime_column=spec['datetime_column'],
        source_column='source',
        predictive_horizon=spec['predictive_horizon'],
        lead=spec['lead'],
        slide=spec['slide'],
        # The previous loaders' rule. It also bounds the profile ladder, which the
        # auto-profile constraint function rejects anything above.
        max_wait_time=math.ceil(min(df.shape[0] for df in frames) / 3),
        train_sources=0.6, val_sources=0.2, test_sources=0.2,
        random_state=SPLIT_RANDOM_STATE,
    )


def build_scania_dataset() -> Dataset:
    """SCANIA Component X from the stacked train/validation/test CSVs.

    Reads them through the data folder's own ``load_frames``, which also checks
    that the three files share one schema. The vehicle split is the one the
    SCANIA authors published, passed as explicit lists, so SPLIT_RANDOM_STATE
    plays no part. events.csv dates each failure at or after the vehicle's last
    readout, so every vehicle is exactly one run-to-failure episode and there is
    no censored tail to keep.
    """
    spec = DATASETS['scania']
    loader_spec = importlib.util.spec_from_file_location(
        'load_scania', os.path.join(spec['folder'], 'load_scania.py'))
    loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader)

    try:
        data, events, splits = loader.load_frames(spec['folder'])
    except FileNotFoundError as error:
        raise SystemExit(str(error))
    sources = splits['sources']

    return Dataset(
        data=data,
        datetime_column=spec['datetime_column'],
        source_column='source',
        event_df=events,
        failure_column=['failure'],
        predictive_horizon=spec['predictive_horizon'],
        lead=spec['lead'],
        slide=spec['slide'],
        # Same rule as build_dataset; one vehicle is one episode.
        max_wait_time=math.ceil(data.groupby('source').size().min() / 3),
        train_sources=[str(source) for source in sources['train']],
        val_sources=[str(source) for source in sources['validation']],
        test_sources=[str(source) for source in sources['test']],
        keep_censored_tail=False,
    )


# --------------------------------------------------------------------------- #
# Grid execution
# --------------------------------------------------------------------------- #
def run_cell(flavor, optimizer, method_name, seed, train_val, args, maximum_profile,
             profile_ladder):
    """One search: one method, one flavor, one backend, one seed.

    One method per call rather than one call per flavor -- an exception inside
    run_experiment aborts its whole method loop, so batching would let a single
    failure cost the other methods their entire budget.
    """
    spec = FLAVORS[flavor]
    method_class = spec['methods'][method_name][0]
    method_space, profile_size, initial_profile_size = cell_space(
        flavor, method_name, maximum_profile, profile_ladder)

    return run_experiment(
        dataset=train_val,
        methods=[method_class],
        param_space_dict_per_method=[method_space],
        method_names=[method_name],
        experiments=[spec['experiment']],
        experiment_names=[f'{args.experiment_prefix} {flavor} {optimizer} seed{seed}'],
        MAX_RUNS=args.max_runs,
        MAX_JOBS=args.max_jobs,
        INITIAL_RANDOM=args.initial_random,
        profile_size=profile_size or profile_ladder,
        initial_profile_size=initial_profile_size or profile_size,
        mlflow_port=MLFLOW_PORT,
        optimization_param=OBJECTIVE,
        optimizer=optimizer,
        random_state=seed,
        debug=False,
        log_best_pipeline=False,
    )


def split_best_params(best_params):
    """Turn one search result back into pinned run_experiment arguments.

    ``best_params`` comes back flattened and prefixed the way the experiment
    logged it, so method axes lose their prefix and the two profile axes travel
    as their own arguments -- only run_experiment may inject those. Every value
    becomes a one-element list: a search space of exactly one configuration.
    """
    method_space, extra = {}, {}
    profile_size = initial_profile_size = None
    for key, value in (best_params or {}).items():
        if key == 'profile_size':
            profile_size = [value]
        elif key == 'initial_profile_size':
            initial_profile_size = [value]
        elif key.startswith('method_'):
            method_space[key[len('method_'):]] = [value]
        else:
            extra[key] = [value]
    return method_space, profile_size, initial_profile_size, extra


def run_test_cell(flavor, optimizer, method_name, seed, test_data, args, best_params,
                  threshold, profile_ladder):
    """Evaluate one search's winning configuration on the held-out test episodes.

    Not a search -- one pinned configuration, one trial -- but it goes through
    run_experiment so the pipeline, constraint function, evaluators and MLflow
    bookkeeping are assembled exactly as they were during the search. Carrying
    the selected threshold over keeps the operating point the one the search
    chose rather than ConstantThresholder's 0.5 default.
    """
    spec = FLAVORS[flavor]
    method_class, _ = spec['methods'][method_name]
    method_space, profile_size, initial_profile_size, extra = split_best_params(best_params)

    if threshold is not None:
        extra.setdefault('thresholder_threshold_value', [threshold])

    return run_experiment(
        dataset=test_data,
        methods=[method_class],
        param_space_dict_per_method=[method_space],
        method_names=[method_name],
        experiments=[spec['experiment']],
        experiment_names=[f'{args.experiment_prefix} {flavor} {optimizer} seed{seed} TEST'],
        MAX_RUNS=2,
        MAX_JOBS=1,
        INITIAL_RANDOM=1,
        profile_size=profile_size or profile_ladder,
        initial_profile_size=initial_profile_size or profile_size,
        additional_parameters=extra,
        mlflow_port=MLFLOW_PORT,
        optimization_param=OBJECTIVE,
        optimizer=TEST_OPTIMIZER,
        random_state=seed,
        debug=True,
        log_best_pipeline=False,
    )


def main(argv=None):
    args = parse_args(argv)

    dataset_handler = build_dataset(args.dataset)

    datasets, test_datasets = {}, {}
    for flavor in args.flavors:
        getter = getattr(dataset_handler, FLAVORS[flavor]['dataset_getter'])
        datasets[flavor], test_datasets[flavor] = getter()

    maximum_profile = dataset_handler.max_wait_time
    profile_ladder = profile_values(maximum_profile)

    print('=' * 78)
    print(f"data set          {args.dataset}  ({DATASETS[args.dataset]['folder']})")
    print(f"episodes          {len(dataset_handler.sources)}  "
          f"train={len(dataset_handler.sources_for_train)} "
          f"val={len(dataset_handler.sources_for_val)} "
          f"test={len(dataset_handler.sources_for_test)}")
    print(f"predictive_horizon {dataset_handler.predictive_horizon}   "
          f"lead {dataset_handler.lead}   slide {dataset_handler.slide}   "
          f"max_wait_time {dataset_handler.max_wait_time}")
    print(f"profile_size      {profile_ladder}")
    print(f"budget            MAX_RUNS={args.max_runs} MAX_JOBS={args.max_jobs} "
          f"INITIAL_RANDOM={args.initial_random}  objective={OBJECTIVE}")

    cells = []
    for seed in SEEDS:
        for flavor in args.flavors:
            available = FLAVORS[flavor]['methods']
            for optimizer in args.optimizers:
                for name in args.methods:
                    if name in available:
                        cells.append((seed, flavor, optimizer, name))

    for flavor in args.flavors:
        skipped = [name for name in args.methods if name not in FLAVORS[flavor]['methods']]
        if skipped:
            print(f"  {flavor}: skipping {', '.join(skipped)} (no {flavor} implementation)")

    print(f"seeds             {', '.join(str(seed) for seed in SEEDS)}")
    print(f"grid              {len(cells)} cell(s), "
          f"<= {len(cells) * args.max_runs} trials in total")
    print('=' * 78)

    if args.dry_run:
        # One line per distinct cell; the seeds only repeat it.
        seen = set()
        for _, flavor, optimizer, name in cells:
            if (flavor, optimizer, name) in seen:
                continue
            seen.add((flavor, optimizer, name))
            full = dict(FLAVORS[flavor]['methods'][name][1](maximum_profile))
            method_dims = len(full)
            if flavor == 'auto_profile':
                full['profile_size'] = profile_ladder
                full['initial_profile_size'] = profile_ladder
            method_space, profile_size, initial_profile_size = cell_space(
                flavor, name, maximum_profile, profile_ladder)
            thinned = dict(method_space)
            if profile_size is not None:
                thinned['profile_size'] = profile_size
                thinned['initial_profile_size'] = initial_profile_size
            print(f"  {flavor:13s} {optimizer:13s} {name:6s} "
                  f"{method_dims} method dim(s) + {len(full) - method_dims} pipeline, "
                  f"space {space_size(full)} -> {space_size(thinned)}")
        return 0

    records = []
    for index, (seed, flavor, optimizer, method_name) in enumerate(cells, start=1):
        header = (f"[{index}/{len(cells)}] seed {seed} :: {flavor} :: {optimizer} "
                  f":: {method_name}")
        print('\n' + '-' * 78 + f"\n{header}\n" + '-' * 78, flush=True)

        started = time.time()
        record = {'seed': seed, 'flavor': flavor, 'optimizer': optimizer, 'method': method_name}
        try:
            results = run_cell(flavor, optimizer, method_name, seed, datasets[flavor],
                               args, maximum_profile, profile_ladder)
        except Exception:
            # One failing cell must not cost the others theirs.
            print(f"!! {header} FAILED:\n{traceback.format_exc()}", flush=True)
            record.update(status='failed', elapsed_s=round(time.time() - started, 1))
            records.append(record)
            continue

        record.update(status='ok', elapsed_s=round(time.time() - started, 1))
        result = results[0] if results else None
        best_params, threshold = None, None
        if isinstance(result, dict):
            best_params = result.get('best_params')
            threshold = result.get('th')
            record['best_objective'] = result.get('best_objective')
            record['threshold'] = threshold

        if best_params:
            print(f"\n  -> test evaluation of {method_name} "
                  f"({flavor}, from {optimizer}, seed {seed})", flush=True)
            try:
                test_results = run_test_cell(flavor, optimizer, method_name, seed,
                                             test_datasets[flavor], args,
                                             best_params, threshold, profile_ladder)
                test_result = test_results[0] if test_results else None
                record['test_status'] = 'ok'
                if isinstance(test_result, dict):
                    record['test_objective'] = test_result.get('best_objective')
                    record['test_threshold'] = test_result.get('th')
            except Exception:
                # The search result is already in hand; losing its test
                # evaluation must not discard it.
                print(f"!! test evaluation FAILED:\n{traceback.format_exc()}", flush=True)
                record['test_status'] = 'failed'

        records.append(record)

    print('\n' + '=' * 78)
    print(f"finished {len(records)} cell(s)")
    ok = [r for r in records if r['status'] == 'ok' and r.get('best_objective') is not None]
    if ok:
        summary = pd.DataFrame(ok)
        columns = {'best_objective': 'val'}
        if 'test_objective' in summary.columns:
            columns['test_objective'] = 'test'
        aggregated = (summary.rename(columns=columns)
                             .groupby(['flavor', 'method', 'optimizer'])[list(columns.values())]
                             .agg(['mean', 'std', 'count'])
                             .sort_values(('val', 'mean'), ascending=False))
        print(f"\n{OBJECTIVE} over {len(SEEDS)} seed(s) "
              "(val = search result, test = same params on test):")
        print(aggregated.to_string())
    failed = [r for r in records if r['status'] == 'failed']
    if failed:
        print(f"\n{len(failed)} cell(s) failed; see the traceback(s) above.")
    return 1 if failed else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', choices=sorted(DATASETS))
    parser.add_argument('--flavors', nargs='+', default=list(FLAVORS), choices=list(FLAVORS))
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument('--optimizers', nargs='+', default=ALL_OPTIMIZERS, choices=ALL_OPTIMIZERS)
    parser.add_argument('--max-runs', type=int, default=200)
    parser.add_argument('--max-jobs', type=int, default=12)
    parser.add_argument('--initial-random', type=int, default=20)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the grid and the search-space sizes, run nothing')
    args = parser.parse_args(argv)
    args.experiment_prefix = f'[{args.dataset}]'
    return args


if __name__ == '__main__':
    sys.exit(main())
