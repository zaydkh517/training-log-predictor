import numpy as np
import pandas as pd

# Heuristic constant: with ~3 fitted points the raw slope gets trusted about
# half; more points -> closer to full trust. Hand-picked, not derived.
SLOPE_SHRINKAGE_K = 3

# Growth ceiling bounds, stated for a ~6-month reference horizon and scaled
# to whatever horizon is actually requested. 
MIN_GROWTH_PCT = 0.05
MAX_GROWTH_PCT = 0.35
EXPERIENCE_K = 30
REF_HORIZON_DAYS = 180  # horizon the research-derived percentages refer to


def experience_growth_cap(n_sessions, span_days):
    """Growth ceiling percentage, scaled by observable experience on this lift.

    Experience is the LARGER of sessions logged and weeks of logged history,
    so a long, sparse log isn't mistaken for a beginner's. Honest limitation:
    training done before the person started logging is invisible here -- a
    10-year lifter with a fresh log still looks novice. Not fixable from log
    data alone.
    """
    experience = max(n_sessions, span_days / 7.0)
    return MIN_GROWTH_PCT + (MAX_GROWTH_PCT - MIN_GROWTH_PCT) * EXPERIENCE_K / (experience + EXPERIENCE_K)

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


def long_term_outlook(e1rm_df, exercise_name, months_ahead=3, recent_window_days=90):

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

    # Shrink the slope toward zero when it's estimated from few recent points.
    # The raw slope is still shown as "your rate" -- only the forward
    # projection uses this more conservative, shrunk version.
    confidence = len(fit_data) / (len(fit_data) + SLOPE_SHRINKAGE_K)
    projection_slope = slope * confidence

    # Uncertainty from how much the RAW session e1RMs scatter around the
    # fitted trend -- not the rolling average's residuals, which are already
    # smoothed and make the band flatteringly narrow. This is a heuristic
    # spread, not a confidence interval, and is labeled as such below.
    raw_yr = fit_data['e1rm'].values.astype(float)
    residual_std = (raw_yr - (slope * Xr + intercept)).std()

    current_anchor = data['rolling_e1rm'].iloc[-1]

    # Never extrapolate the linear trend further than the span of data it was
    # fit on: projected growth saturates smoothly as the horizon exceeds the
    # observed window, instead of running linearly into the void.
    horizon_days = months_ahead * 30
    fit_span_days = float(Xr.max() - Xr.min())
    if fit_span_days > 0:
        effective_days = fit_span_days * (1 - np.exp(-horizon_days / fit_span_days))
    else:
        effective_days = 0.0
    point_estimate = current_anchor + projection_slope * effective_days

    # Experience-scaled ceiling, scaled from its 6-month reference figures to
    # the actual horizon requested.
    n_sessions = len(data)
    span_days = (data['date'].max() - data['date'].min()).days
    growth_cap_pct = experience_growth_cap(n_sessions, span_days) * (horizon_days / REF_HORIZON_DAYS)
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
        'range_basis': 'typical session-to-session variability around the recent trend -- a heuristic spread, not a statistical confidence interval',
    }
