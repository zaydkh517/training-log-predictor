import numpy as np
import pandas as pd


def check_plateau(e1rm_df, exercise_name, threshold=0.02, lookback_sessions=4):

    data = e1rm_df[e1rm_df['exercise'] == exercise_name].sort_values('date')

    if len(data) < lookback_sessions + 1:
        return {'exercise': exercise_name, 'error': 'not enough sessions to assess yet'}

    recent_rolling = data['rolling_e1rm'].iloc[-1]
    past_rolling = data['rolling_e1rm'].iloc[-(lookback_sessions + 1)]
    percent_change = (recent_rolling - past_rolling) / past_rolling
    is_plateau = percent_change < threshold

    return {
        'exercise': exercise_name,
        'rolling_e1rm_lookback_sessions_ago': float(round(past_rolling, 2)),
        'rolling_e1rm_now': float(round(recent_rolling, 2)),
        'percent_change': float(round(percent_change, 4)),
        'is_plateau': bool(is_plateau),
    }


def long_term_outlook(e1rm_df, exercise_name, months_ahead=6, recent_window_days=90):

    data = e1rm_df[e1rm_df['exercise'] == exercise_name].sort_values('date').copy()
    if len(data) < 4:
        return {'exercise': exercise_name, 'error': 'not enough sessions for a trend line yet'}

    data['days_since_start'] = (data['date'] - data['date'].min()).dt.days

    cutoff = data['date'].max() - pd.Timedelta(days=recent_window_days)
    recent = data[data['date'] >= cutoff]
    fit_data = recent if len(recent) >= 4 else data

    Xr = fit_data['days_since_start'].values.astype(float)
    yr = fit_data['rolling_e1rm'].values.astype(float)
    slope, intercept = np.polyfit(Xr, yr, 1)  # lbs/day, from the recent pace

    # uncertainty band from the FULL history's swings around this rate-based line
    Xf = data['days_since_start'].values.astype(float)
    yf = data['rolling_e1rm'].values.astype(float)
    residuals_full = yf - (slope * Xf + intercept)
    residual_std = residuals_full.std()

    current_anchor = data['rolling_e1rm'].iloc[-1]
    point_estimate = current_anchor + slope * (months_ahead * 30)
    low = point_estimate - 1.5 * residual_std
    high = point_estimate + 1.5 * residual_std

    return {
        'exercise': exercise_name,
        'raw_last_session_e1rm_lbs': float(data['e1rm'].iloc[-1]),
        'current_smoothed_e1rm_lbs': float(round(current_anchor, 1)),
        'your_rate_lbs_per_week': float(round(slope * 7, 2)),
        'rate_window': f'last {recent_window_days} days' if len(recent) >= 4 else 'full history (not enough recent data)',
        f'outlook_{months_ahead}mo_range_lbs': (float(round(low, 1)), float(round(high, 1))),
    }
