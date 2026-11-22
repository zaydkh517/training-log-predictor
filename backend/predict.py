import numpy as np
import pandas as pd

from forecast import forecast_lift, load_model

RATE_WINDOW_DAYS = 90  # window for the displayed lbs/week trend

# growth ceilings: novices gain ~20-40% per 6 months, trained lifters ~2-6%/year
MIN_GROWTH_PCT = 0.05
MAX_GROWTH_PCT = 0.35
EXPERIENCE_K = 30
REF_HORIZON_DAYS = 180  # the caps above are stated per ~6 months

PLATEAU_WINDOW_DAYS = 28
PLATEAU_FRACTION = 0.5          # stalled = under half the gain expected for this training age
MIN_PLATEAU_THRESHOLD = 0.005   # below +0.5% is noise for anyone
DECLINE_THRESHOLD = -0.02       # down 2%+ over the window
INACTIVE_AFTER_DAYS = 45        # not trained in 6+ weeks: don't call it current


def experience_growth_cap(n_sessions, span_days):
    """Growth cap for this lift, from sessions logged or weeks of history (whichever is larger)."""
    # limitation: training done before the person started logging is invisible
    experience = max(n_sessions, span_days / 7.0)
    return MIN_GROWTH_PCT + (MAX_GROWTH_PCT - MIN_GROWTH_PCT) * EXPERIENCE_K / (experience + EXPERIENCE_K)


def days_inactive(e1rm_df, history):
    """Days between this lift's last session and the newest session anywhere in the log.
    Measured against the log, not the wall clock, so an old export isn't all 'inactive'."""
    return int((e1rm_df['date'].max() - history['date'].max()).days)


def lift_history(e1rm_df, exercise_name):
    """(all sessions of the lift, sessions on the machine it was last done on).

    Trends only make sense within one machine; training age counts all of them.
    """
    history = e1rm_df[e1rm_df['exercise'] == exercise_name].sort_values('date')
    if 'machine' not in history.columns:
        return history, history
    return history, history[history['machine'] == history['machine'].iloc[-1]]


def check_plateau(e1rm_df, exercise_name, window_days=PLATEAU_WINDOW_DAYS):
    full, history = lift_history(e1rm_df, exercise_name)
    base = {'exercise': exercise_name, 'window_days': window_days}

    idle = days_inactive(e1rm_df, full)
    if idle > INACTIVE_AFTER_DAYS:
        return {**base, 'status': 'inactive', 'is_plateau': False, 'days_since_last_session': idle}

    # compare against the last session at least `window_days` before the most recent one:
    # a fixed span of time, not a fixed number of sessions
    last_date = history['date'].iloc[-1]
    past = history[history['date'] <= last_date - pd.Timedelta(days=window_days)]
    if past.empty:
        if len(history) < len(full):
            return {**base, 'error': f'new machine -- needs {window_days} days on it to assess'}
        return {**base, 'error': f'needs at least {window_days} days of history to assess'}

    recent_rolling = history['rolling_e1rm'].iloc[-1]
    past_rolling = past['rolling_e1rm'].iloc[-1]
    percent_change = (recent_rolling - past_rolling) / past_rolling

    # "stalled" means well under the gain expected at this training age, not a fixed 2%;
    # training age counts every machine, since the lifter's experience carries over
    span_days = (last_date - full['date'].iloc[0]).days
    expected = experience_growth_cap(len(full), span_days) * (window_days / REF_HORIZON_DAYS)
    threshold = max(MIN_PLATEAU_THRESHOLD, PLATEAU_FRACTION * expected)

    if percent_change <= DECLINE_THRESHOLD:
        status = 'declining'
    elif percent_change < threshold:
        status = 'plateau'
    else:
        status = 'progressing'

    return {
        **base,
        'status': status,
        'window_start': past['date'].iloc[-1].strftime('%Y-%m-%d'),
        'rolling_e1rm_window_start': float(round(past_rolling, 2)),
        'rolling_e1rm_now': float(round(recent_rolling, 2)),
        'percent_change': float(round(percent_change, 4)),
        'threshold_pct': float(round(threshold, 4)),
        'is_plateau': status in ('plateau', 'declining'),
        'days_since_last_session': idle,
    }


def long_term_outlook(e1rm_df, exercise_name, interval_scale=1.0, model=None):
    """3-month outlook from the learned forecaster (see forecast.py), for the machine the
    lift was last done on. `interval_scale` widens the 80% interval for this lifter."""
    full, history = lift_history(e1rm_df, exercise_name)
    if len(full) < 4:
        return {'exercise': exercise_name, 'error': 'not enough sessions for a trend line yet'}

    idle = days_inactive(e1rm_df, full)
    current_level = history['rolling_e1rm'].iloc[-1]
    if idle > INACTIVE_AFTER_DAYS:
        # still report the last level, but don't forecast a lift that isn't being trained
        return {
            'exercise': exercise_name,
            'current_smoothed_e1rm_lbs': float(round(current_level, 1)),
            'error': f'not trained in {idle} days -- no outlook',
        }
    forecast = forecast_lift(model or load_model(), full, history, interval_scale)
    if forecast is None:
        return {
            'exercise': exercise_name,
            'current_smoothed_e1rm_lbs': float(round(current_level, 1)),
            'error': 'new machine -- not enough sessions on it for a trend yet',
        }
    low, median, high = forecast

    # the displayed rate: a plain trend over the last 90 days on this machine
    cutoff = history['date'].max() - pd.Timedelta(days=RATE_WINDOW_DAYS)
    recent = history[history['date'] >= cutoff]
    fit_data = recent if len(recent) >= 4 else history
    days = (fit_data['date'] - fit_data['date'].min()).dt.days.to_numpy(dtype=float)
    slope = np.polyfit(days, fit_data['rolling_e1rm'].to_numpy(dtype=float), 1)[0] if days.max() > 0 else 0.0

    return {
        'exercise': exercise_name,
        'raw_last_session_e1rm_lbs': float(history['e1rm'].iloc[-1]),
        'current_smoothed_e1rm_lbs': float(round(current_level, 1)),
        'your_rate_lbs_per_week': float(round(slope * 7, 2)),
        'rate_window': f'last {RATE_WINDOW_DAYS} days' if len(recent) >= 4 else 'full history (not enough recent data)',
        'outlook_3mo_range_lbs': (float(round(low, 1)), float(round(high, 1))),
        'outlook_3mo_median_lbs': float(round(median, 1)),
        'method': 'learned',
        'interval_scale': float(round(interval_scale, 2)),
        'range_basis': (
            '80% prediction interval from a model trained on 1.2M meet-to-meet changes '
            '(OpenPowerlifting), widened to match how well it has forecast your own history'
        ),
    }
