import pandas as pd

from predict import (
    check_plateau,
    long_term_outlook,
    experience_growth_cap,
    REF_HORIZON_DAYS,
)


def _e1rm_history(rolling_values, start='2026-01-01', freq='7D'):
    """Minimal e1rm_df for one exercise: one session per week, with e1rm and
    rolling_e1rm set directly to the values given."""
    dates = pd.date_range(start, periods=len(rolling_values), freq=freq)
    return pd.DataFrame({
        'date': dates,
        'exercise': ['Bench Press (Barbell)'] * len(rolling_values),
        'e1rm': [float(v) for v in rolling_values],
        'rolling_e1rm': [float(v) for v in rolling_values],
    })


def test_plateau_needs_enough_sessions():
    df = _e1rm_history([100, 101, 102])  # 3 sessions < lookback 4 + 1
    result = check_plateau(df, 'Bench Press (Barbell)')
    assert 'error' in result


def test_flat_trend_is_plateau():
    df = _e1rm_history([150] * 6)
    result = check_plateau(df, 'Bench Press (Barbell)')
    assert result['is_plateau'] is True


def test_growing_trend_is_not_plateau():
    df = _e1rm_history([100, 105, 110, 115, 120, 125])
    result = check_plateau(df, 'Bench Press (Barbell)')
    assert result['is_plateau'] is False


def test_outlook_respects_growth_ceiling():
    # absurdly steep recent progress: +10 lbs/week, every week
    rolling = [100, 110, 120, 130, 140, 150]
    df = _e1rm_history(rolling)

    result = long_term_outlook(df, 'Bench Press (Barbell)')  # default: 3 months

    # recompute the ceiling exactly as predict.py defines it:
    # 6 weekly sessions -> 35 days of history, cap scaled from the 6-month
    # reference figures down to the 90-day horizon
    span_days = 35
    growth_cap = experience_growth_cap(len(rolling), span_days) * (90 / REF_HORIZON_DAYS)
    ceiling = 150 * (1 + growth_cap)

    low, high = result['outlook_3mo_range_lbs']
    assert high <= ceiling + 1e-6
    assert low >= 150  # never projected below current level


def test_absurd_horizon_is_bounded_by_data_span():
    # Ask for a 5-YEAR outlook from 35 days of data. Saturation must stop the
    # linear trend from extrapolating past the growth implied by the window
    # it was fit on (slope 10/7 lbs/day over a 35-day span).
    rolling = [100, 110, 120, 130, 140, 150]
    df = _e1rm_history(rolling)

    result = long_term_outlook(df, 'Bench Press (Barbell)', months_ahead=60)

    low, high = result['outlook_60mo_range_lbs']
    assert high <= 150 + (10 / 7) * 35 + 1e-6