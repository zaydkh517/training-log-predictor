"""3-month e1RM forecasting with a model learned from OpenPowerlifting.

The model is three gradient-boosted quantile regressors (10th, 50th, 90th percentile)
of log change in a lift over the coming months. It was trained on ~1.2M meet-to-meet
changes from raw lifters (see train_forecast_model.py), using only inputs a workout
log can also supply -- no lift type, bodyweight or absolute weight -- so the same
model applies to any exercise:

    horizon     days ahead being forecast
    train_age   days since the lift was first logged (a proxy for training age)
    prev_gap    days back to the comparison point for the recent rate
    prev_rate   log change per 30 days since that point (NaN when there isn't one)

Meet lifters are more advanced than the average gym log, and a smoothed gym e1RM
isn't a meet max, so the raw 80% interval is too narrow on gym data. Each upload is
backtested against itself and the interval is widened (never narrowed, never
re-centred) to match how well the model has forecast that lifter before.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURES = ['horizon', 'train_age', 'prev_gap', 'prev_rate']
QUANTILES = (0.1, 0.5, 0.9)
HORIZON_DAYS = 90
MIN_HORIZON_DAYS, MAX_HORIZON_DAYS = 60, 200   # the meet-to-meet gaps it was trained on
RATE_LOOKBACK_DAYS = 90                        # recent rate: vs the session nearest 90 days back...
MIN_RATE_GAP_DAYS = 30                         # ...but at least 30 days back
MIN_SESSIONS = 4                               # sessions on a machine before it gets forecast
CALIBRATION_SHRINKAGE = 20                     # with k backtests, trust k/(k+20) of the widening

MODEL_PATH = Path(__file__).with_name('forecast_model.joblib')
_loaded = {}


def load_model(path=MODEL_PATH):
    """{quantile: fitted regressor}, cached per path."""
    if path not in _loaded:
        _loaded[path] = joblib.load(path)
    return _loaded[path]


def features_at(dates, rolling, i, first_date, horizon=HORIZON_DAYS):
    """Model inputs for a forecast made at session `i` of one machine's history.

    `dates` and `rolling` are that machine's session dates and rolling e1RMs in date
    order; `first_date` is the lift's first session on any machine.
    """
    now = dates[i]
    earlier = np.flatnonzero(dates[:i] <= now - np.timedelta64(MIN_RATE_GAP_DAYS, 'D'))
    if earlier.size:
        target = now - np.timedelta64(RATE_LOOKBACK_DAYS, 'D')
        j = earlier[np.argmin(np.abs(dates[earlier] - target))]
        gap = (now - dates[j]) / np.timedelta64(1, 'D')
        rate = np.log(rolling[i] / rolling[j]) * 30 / max(gap, 7)
    else:
        gap, rate = np.nan, np.nan
    return {
        'horizon': float(horizon),
        'train_age': (now - first_date) / np.timedelta64(1, 'D'),
        'prev_gap': gap,
        'prev_rate': rate,
    }


def predict_quantiles(model, rows):
    """Log-change quantiles, one row per forecast, kept in order (q10 <= q50 <= q90)."""
    X = pd.DataFrame(rows, columns=FEATURES)
    preds = np.column_stack([model[q].predict(X) for q in QUANTILES])
    return np.sort(preds, axis=1)


def backtest(e1rm_df):
    """Every past forecast whose outcome is already in the log.

    One row per (lift, machine, session) with at least MIN_SESSIONS prior sessions and
    a later session on the same machine 60-200 days out. `y` is the realized log change
    in rolling e1RM; the features are what the forecast would have seen at the time.
    """
    rows = []
    first_dates = e1rm_df.groupby('exercise')['date'].min()
    machine_col = 'machine' if 'machine' in e1rm_df.columns else None
    groups = e1rm_df.groupby(['exercise', machine_col] if machine_col else 'exercise')
    for key, sessions in groups:
        exercise = key[0] if machine_col else key
        sessions = sessions.sort_values('date')
        dates = sessions['date'].to_numpy()
        rolling = sessions['rolling_e1rm'].to_numpy(dtype=float)
        first = np.datetime64(first_dates[exercise])
        for i in range(MIN_SESSIONS, len(dates)):
            lo = dates[i] + np.timedelta64(MIN_HORIZON_DAYS, 'D')
            k = np.searchsorted(dates, lo)
            if k >= len(dates) or dates[k] > dates[i] + np.timedelta64(MAX_HORIZON_DAYS, 'D'):
                continue
            horizon = (dates[k] - dates[i]) / np.timedelta64(1, 'D')
            row = features_at(dates, rolling, i, first, horizon=horizon)
            row['y'] = float(np.log(rolling[k] / rolling[i]))
            rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES + ['y'])


def interval_scale(model, e1rm_df):
    """How much to widen the 80% interval for this lifter, from their own backtests.

    Returns (scale >= 1, number of backtests). With no history it's 1 (the meet-data
    interval); with k backtests it moves k/(k+20) of the way to the width that would
    have covered 80% of this lifter's past outcomes.
    """
    past = backtest(e1rm_df)
    k = len(past)
    if k == 0:
        return 1.0, 0
    q = predict_quantiles(model, past[FEATURES])
    mid, half = (q[:, 0] + q[:, 2]) / 2, np.maximum((q[:, 2] - q[:, 0]) / 2, 1e-3)
    needed = float(np.quantile(np.abs(past['y'].to_numpy() - mid) / half, 0.8))
    weight = k / (k + CALIBRATION_SHRINKAGE)
    return max(1.0, 1 + weight * (needed - 1)), k


def forecast_lift(model, full, current, scale=1.0):
    """3-month forecast for one lift from its current machine's sessions.

    Returns (low, median, high) in the lift's units, as an 80% interval around the
    model's median, or None when the machine has too few sessions.
    """
    if len(current) < MIN_SESSIONS:
        return None
    dates = current['date'].to_numpy()
    rolling = current['rolling_e1rm'].to_numpy(dtype=float)
    row = features_at(dates, rolling, len(dates) - 1, np.datetime64(full['date'].min()))
    q10, q50, q90 = predict_quantiles(model, [row])[0]
    mid, half = (q10 + q90) / 2, (q90 - q10) / 2 * scale
    level = rolling[-1]
    return level * np.exp(mid - half), level * np.exp(q50), level * np.exp(mid + half)
