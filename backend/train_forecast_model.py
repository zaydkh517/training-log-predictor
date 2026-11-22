"""Train and evaluate the 3-month e1RM forecaster; writes forecast_model.joblib.

Training data: the OpenPowerlifting bulk CSV (public domain), unzipped anywhere under
data/opl/ (gitignored):
    https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip

Each example is one raw lifter's best squat, bench or deadlift at a meet, paired with
their best at their next meet 60-200 days later. The inputs are only what a workout
log can also supply (see forecast.FEATURES), so the model applies to any exercise.

Evaluation:
  1. OpenPowerlifting: lifters never seen in training, forecasts made in 2023 or later
     (training targets all fall before 2023), against simple baselines and the
     heuristic this model replaced.
  2. Gym logs, if any are in data/: the same backtest the API runs on each upload.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import build_dataset
from forecast import (
    FEATURES, MAX_HORIZON_DAYS, MIN_HORIZON_DAYS, MODEL_PATH, QUANTILES, backtest, predict_quantiles,
)
from predict import experience_growth_cap
from train_rep_model import DATA, export_kind

LIFT_COLUMNS = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']
TEST_FROM = pd.Timestamp('2023-01-01')
MAX_ABS_CHANGE = np.log(1.5)   # +/-50% in a few months is a data error or two people merged


def build_pairs(csv_path):
    raw = pd.read_csv(csv_path, usecols=['Name', 'Equipment', 'Date', *LIFT_COLUMNS],
                      dtype={'Name': str, 'Equipment': 'category'})
    raw = raw[raw['Equipment'] == 'Raw']   # gym logs are raw; equipped lifts aren't comparable
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')

    frames = []
    for col in LIFT_COLUMNS:
        lift = raw[['Name', 'Date', col]].rename(columns={col: 'best'})
        lift = lift[lift['best'] > 0].dropna(subset=['Date'])   # failed lifts are stored negative
        # one value per lifter per day (the same meet entered in several divisions)
        lift = lift.sort_values('best').drop_duplicates(['Name', 'Date'], keep='last')
        lift['key'] = lift['Name'] + '|' + col
        frames.append(lift)
    obs = pd.concat(frames, ignore_index=True).sort_values(['key', 'Date']).reset_index(drop=True)
    by_lift = obs.groupby('key')
    obs['first_date'] = by_lift['Date'].transform('min')
    obs['prev_date'] = by_lift['Date'].shift(1)
    obs['prev_best'] = by_lift['best'].shift(1)
    obs['n_prior'] = by_lift.cumcount()

    # target: the first later meet at least MIN_HORIZON_DAYS out, if within MAX_HORIZON_DAYS
    targets = obs[['key', 'Date', 'best']].rename(columns={'Date': 't_date', 'best': 't_best'})
    obs['probe'] = obs['Date'] + pd.Timedelta(days=MIN_HORIZON_DAYS)
    pairs = pd.merge_asof(obs.sort_values('probe'), targets.sort_values('t_date'),
                          left_on='probe', right_on='t_date', by='key', direction='forward')
    pairs = pairs.dropna(subset=['t_date'])
    pairs['horizon'] = (pairs['t_date'] - pairs['Date']).dt.days.astype(float)
    pairs = pairs[pairs['horizon'] <= MAX_HORIZON_DAYS].copy()

    pairs['y'] = np.log(pairs['t_best'] / pairs['best'])
    pairs['train_age'] = (pairs['Date'] - pairs['first_date']).dt.days.astype(float)
    pairs['prev_gap'] = (pairs['Date'] - pairs['prev_date']).dt.days.astype(float)
    pairs['prev_rate'] = np.log(pairs['best'] / pairs['prev_best']) * 30 / pairs['prev_gap'].clip(lower=7)
    return pairs[pairs['y'].abs() < MAX_ABS_CHANGE].reset_index(drop=True)


def fit(train):
    model = {}
    for q in QUANTILES:
        model[q] = HistGradientBoostingRegressor(
            loss='quantile', quantile=q, max_iter=300, learning_rate=0.08,
            max_leaf_nodes=31, min_samples_leaf=200, random_state=0,
        ).fit(train[FEATURES], train['y'])
    return model


def replaced_heuristic(d):
    """The pre-model outlook logic, adapted to two meet points: a trust-shrunk, damped
    slope clamped by the experience-scaled growth cap."""
    slope = np.nan_to_num(d['prev_rate']) / 30
    span = d['prev_gap'].fillna(0).to_numpy()
    effective = np.where(span > 0, span * (1 - np.exp(-d['horizon'] / np.maximum(span, 1))), 0.0)
    point = slope * (2 / 5) * effective
    cap = np.array([experience_growth_cap(n + 1, a) for n, a in zip(d['n_prior'], d['train_age'])])
    cap = cap * d['horizon'] / 180
    return np.where(point >= 0, np.minimum(point, np.log1p(cap)), np.maximum(point, np.log1p(-cap)))


def scores(y, q, extra):
    mae = lambda p: float(np.mean(np.abs(y - p))) * 100
    row = {name: mae(pred) for name, pred in extra.items()}
    row['Learned model (median)'] = mae(q[:, 1])
    row['80% interval coverage'] = float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))) * 100
    return row


def gym_logs():
    """e1RM tables for each lifter in data/ (same layout as train_rep_model.py)."""
    tables = []
    strong, hevy = DATA / 'strong_export.csv', DATA / 'hevy_export.csv'
    if strong.exists() or hevy.exists():
        tables.append(build_dataset(strong if strong.exists() else None, hevy if hevy.exists() else None)['e1rm_df'])
    for path in sorted((DATA / 'lifters').glob('*.csv')):
        kind = export_kind(path)
        tables.append(build_dataset(**{f'{kind}_path': path})['e1rm_df'])
    return tables


def main():
    csvs = sorted(Path('data/opl').rglob('openpowerlifting-*.csv'))
    if not csvs:
        raise SystemExit('download the OpenPowerlifting CSV into data/opl/ first (see the docstring)')
    start = time.time()
    pairs = build_pairs(csvs[-1])
    print(f'{len(pairs):,} meet-to-meet pairs from {pairs["Name"].nunique():,} raw lifters '
          f'({time.time() - start:.0f}s)\n')

    # held-out lifters, forecasts from 2023 on; training targets strictly before 2023
    names = pairs['Name'].unique()
    held_out = set(np.random.default_rng(0).choice(names, size=len(names) // 5, replace=False))
    is_held = pairs['Name'].isin(held_out)
    train = pairs[~is_held & (pairs['t_date'] < TEST_FROM)]
    test = pairs[is_held & (pairs['Date'] >= TEST_FROM) & pairs['prev_rate'].notna()]
    model = fit(train)
    y = test['y'].to_numpy()
    opl = scores(y, predict_quantiles(model, test[FEATURES]), {
        'No change': 0.0,
        'Median change in training data': float(np.median(train['y'])),
        'Replaced heuristic': replaced_heuristic(test),
    })

    columns = {'OpenPowerlifting (held-out lifters, 2023+)': (opl, len(test))}
    for i, table in enumerate(gym_logs()):
        past = backtest(table)
        if len(past) < 20:
            continue
        columns[f'Gym log, lifter_{i}'] = (scores(past['y'].to_numpy(), predict_quantiles(model, past[FEATURES]),
                                                  {'No change': 0.0}), len(past))

    print('MAE in % of e1RM over the horizon (lower is better):\n')
    print('| | ' + ' | '.join(f'{k} (n={n:,})' for k, (_, n) in columns.items()) + ' |')
    print('|---|' + '---|' * len(columns))
    rows = list(opl)
    for r in rows:
        cells = [f'{c[r]:.2f}' + ('%' if 'coverage' in r else '') if r in c else '' for c, _ in columns.values()]
        print(f'| {r} | ' + ' | '.join(cells) + ' |')

    final = fit(pairs)   # the shipped model learns from every pair
    joblib.dump(final, MODEL_PATH, compress=3)
    print(f'\nSaved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB), trained on {len(pairs):,} pairs')


if __name__ == '__main__':
    main()
