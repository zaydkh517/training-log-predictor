import numpy as np
import pandas as pd

SLOPE_SHRINKAGE_K = 3

# How far a 6-month outlook is allowed to grow, scaled by training experience
# on that specific lift (sessions logged for it -- not a cross-dataset
# comparison, just this person's own history). Novices get closer to
# MAX_GROWTH_PCT, well-established lifts decay toward MIN_GROWTH_PCT.
# Grounded in real strength-training research: novices commonly see 20-40%
# gains over a period like this, well-trained lifters more like 2-6%/year.
MIN_GROWTH_PCT = 0.05
MAX_GROWTH_PCT = 0.35
EXPERIENCE_K = 30

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
    
    # Shrink the slope toward zero when it's estimated from few recent points . The raw slope is still shown as "your rate"
    # and used for the uncertainty calc below -- only the forward projection
    # uses this more conservative, shrunk version.
    confidence = len(fit_data) / (len(fit_data) + SLOPE_SHRINKAGE_K)
    projection_slope = slope * confidence

    # uncertainty band from how much the RECENT window actually wiggled around
    # this fitted line
    residuals_recent = yr - (slope * Xr + intercept)
    residual_std = residuals_recent.std()

    current_anchor = data['rolling_e1rm'].iloc[-1]
    point_estimate = current_anchor + projection_slope * (months_ahead * 30)

    # Experience-scaled ceiling: fewer sessions logged on this lift -> more
    # room to grow (novice-tier); many sessions -> tighter ceiling (advanced-tier).
    n_sessions = len(data)
    growth_cap_pct = MIN_GROWTH_PCT + (MAX_GROWTH_PCT - MIN_GROWTH_PCT) * EXPERIENCE_K / (n_sessions + EXPERIENCE_K)
    ceiling = current_anchor * (1 + growth_cap_pct)
    floor = current_anchor  # never project below where you already are

    # Clamp the CENTER of the range first, then rebuild the +/- uncertainty
    # band around that clamped center. Clamping low/high independently would
    # collapse the whole band to a single point whenever the point estimate
    # overshoots the ceiling -- this keeps a real, honest band width instead.
    center = min(max(point_estimate, floor), ceiling)
    low = max(center - 1.5 * residual_std, floor)
    high = min(center + 1.5 * residual_std, ceiling)

    return {
        'exercise': exercise_name,
        'raw_last_session_e1rm_lbs': float(data['e1rm'].iloc[-1]),
        'current_smoothed_e1rm_lbs': float(round(current_anchor, 1)),
        'your_rate_lbs_per_week': float(round(slope * 7, 2)),
        'rate_window': f'last {recent_window_days} days' if len(recent) >= 4 else 'full history (not enough recent data)',
        f'outlook_{months_ahead}mo_range_lbs': (float(round(low, 1)), float(round(high, 1))),
    }
