import numpy as np
import pandas as pd

SLOPE_SHRINKAGE_K = 3  # fewer fitted points -> trust less of the slope

# growth ceilings: novices gain ~20-40% per 6 months, trained lifters ~2-6%/year
MIN_GROWTH_PCT = 0.05
MAX_GROWTH_PCT = 0.35
EXPERIENCE_K = 30
REF_HORIZON_DAYS = 180  # the caps above are stated per ~6 months


def experience_growth_cap(n_sessions, span_days):
    """Growth cap for this lift, from sessions logged or weeks of history (whichever is larger)."""
    # limitation: training done before the person started logging is invisible
    experience = max(n_sessions, span_days / 7.0)
    return MIN_GROWTH_PCT + (MAX_GROWTH_PCT - MIN_GROWTH_PCT) * EXPERIENCE_K / (experience + EXPERIENCE_K)


def check_plateau(e1rm_df, exercise_name, threshold=0.02, lookback_sessions=4):
    history = e1rm_df[e1rm_df['exercise'] == exercise_name].sort_values('date')

    if len(history) < lookback_sessions + 1:
        return {'exercise': exercise_name, 'error': 'not enough sessions to assess yet'}

    recent_rolling = history['rolling_e1rm'].iloc[-1]
    past_rolling = history['rolling_e1rm'].iloc[-(lookback_sessions + 1)]
    percent_change = (recent_rolling - past_rolling) / past_rolling

    return {
        'exercise': exercise_name,
        'rolling_e1rm_lookback_sessions_ago': float(round(past_rolling, 2)),
        'rolling_e1rm_now': float(round(recent_rolling, 2)),
        'percent_change': float(round(percent_change, 4)),
        'is_plateau': bool(percent_change < threshold),
    }


def long_term_outlook(e1rm_df, exercise_name, months_ahead=3, recent_window_days=90):
    history = e1rm_df[e1rm_df['exercise'] == exercise_name].sort_values('date').copy()
    if len(history) < 4:
        return {'exercise': exercise_name, 'error': 'not enough sessions for a trend line yet'}

    history['days_since_start'] = (history['date'] - history['date'].min()).dt.days

    cutoff = history['date'].max() - pd.Timedelta(days=recent_window_days)
    recent = history[history['date'] >= cutoff]
    fit_data = recent if len(recent) >= 4 else history

    fit_days = fit_data['days_since_start'].values.astype(float)
    fit_rolling = fit_data['rolling_e1rm'].values.astype(float)
    slope, intercept = np.polyfit(fit_days, fit_rolling, 1)  # lbs/day

    # shrink the slope when it was fit on few points
    slope_trust = len(fit_data) / (len(fit_data) + SLOPE_SHRINKAGE_K)
    projection_slope = slope * slope_trust

    # band width from raw session scatter (rolling values are pre-smoothed)
    fit_raw = fit_data['e1rm'].values.astype(float)
    residual_std = (fit_raw - (slope * fit_days + intercept)).std()

    current_level = history['rolling_e1rm'].iloc[-1]

    # projected growth levels off; never extrapolates past the span it was fit on
    horizon_days = months_ahead * 30
    fit_span_days = float(fit_days.max() - fit_days.min())
    if fit_span_days > 0:
        effective_days = fit_span_days * (1 - np.exp(-horizon_days / fit_span_days))
    else:
        effective_days = 0.0
    point_estimate = current_level + projection_slope * effective_days

    # experience cap, scaled from its 6-month reference to this horizon
    span_days = (history['date'].max() - history['date'].min()).days
    growth_cap_pct = experience_growth_cap(len(history), span_days) * (horizon_days / REF_HORIZON_DAYS)
    ceiling = current_level * (1 + growth_cap_pct)
    floor = current_level  # never project below today

    # clamp the center, then rebuild the band around it so it can't collapse to a point
    center = min(max(point_estimate, floor), ceiling)
    low = max(center - 1.5 * residual_std, floor)
    high = min(center + 1.5 * residual_std, ceiling)

    return {
        'exercise': exercise_name,
        'raw_last_session_e1rm_lbs': float(history['e1rm'].iloc[-1]),
        'current_smoothed_e1rm_lbs': float(round(current_level, 1)),
        'your_rate_lbs_per_week': float(round(slope * 7, 2)),
        'rate_window': f'last {recent_window_days} days' if len(recent) >= 4 else 'full history (not enough recent data)',
        f'outlook_{months_ahead}mo_range_lbs': (float(round(low, 1)), float(round(high, 1))),
        'range_basis': 'typical session-to-session variability around the recent trend -- a heuristic spread, not a statistical confidence interval',
    }
