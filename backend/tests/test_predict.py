import pandas as pd

from predict import (
    check_plateau,
    long_term_outlook,
    experience_growth_cap,
    REF_HORIZON_DAYS,
)


def _e1rm_history(rolling_values, start='2026-01-01', freq='7D'):
    """Minimal one-exercise e1rm_df: one session per week."""
    dates = pd.date_range(start, periods=len(rolling_values), freq=freq)
    return pd.DataFrame({
        'date': dates,
        'exercise': ['Bench Press (Barbell)'] * len(rolling_values),
        'e1rm': [float(v) for v in rolling_values],
        'rolling_e1rm': [float(v) for v in rolling_values],
    })


def test_plateau_needs_enough_history():
    df = _e1rm_history([100, 101, 102])  # 14 days < the 28-day window
    result = check_plateau(df, 'Bench Press (Barbell)')
    assert 'error' in result


def test_flat_trend_is_plateau():
    df = _e1rm_history([150] * 6)
    result = check_plateau(df, 'Bench Press (Barbell)')
    assert result['is_plateau'] is True
    assert result['status'] == 'plateau'


def test_growing_trend_is_not_plateau():
    df = _e1rm_history([100, 105, 110, 115, 120, 125])
    result = check_plateau(df, 'Bench Press (Barbell)')
    assert result['is_plateau'] is False


def test_outlook_respects_growth_ceiling():
    # absurdly steep recent progress: +10 lbs/week, every week
    rolling = [100, 110, 120, 130, 140, 150]
    df = _e1rm_history(rolling)

    result = long_term_outlook(df, 'Bench Press (Barbell)')  # default: 3 months

    # recompute the ceiling exactly as predict.py does (6 sessions, 35-day span)
    span_days = 35
    growth_cap = experience_growth_cap(len(rolling), span_days) * (90 / REF_HORIZON_DAYS)
    ceiling = 150 * (1 + growth_cap)

    low, high = result['outlook_3mo_range_lbs']
    assert high <= ceiling + 1e-6
    assert low >= 150  # a rising trend is never projected below current level


def test_absurd_horizon_is_bounded_by_data_span():
    # a 5-year ask can't outgrow the 35-day fitted span (slope 10/7 lbs/day)
    rolling = [100, 110, 120, 130, 140, 150]
    df = _e1rm_history(rolling)

    result = long_term_outlook(df, 'Bench Press (Barbell)', months_ahead=60)

    low, high = result['outlook_60mo_range_lbs']
    assert high <= 150 + (10 / 7) * 35 + 1e-6

def test_plateau_window_is_time_not_sessions():
    # same +6% over 4 weeks, logged 2x/week vs 1x/week: the call must not depend on frequency
    weekly = _e1rm_history([100, 101.5, 103, 104.5, 106], freq='7D')
    twice_weekly = _e1rm_history([100 + 0.75 * i for i in range(9)], freq='3.5D')
    a = check_plateau(weekly, 'Bench Press (Barbell)')
    b = check_plateau(twice_weekly, 'Bench Press (Barbell)')
    assert a['status'] == b['status'] == 'progressing'
    assert abs(a['percent_change'] - b['percent_change']) < 0.005


def test_experienced_lifter_slow_gain_is_not_plateau():
    # two years of weekly sessions: +1.2% a month is real progress at that training age,
    # though the old fixed 2% rule called it a plateau
    values = [200 * (1.012 ** (i / 4.3)) for i in range(104)]
    result = check_plateau(_e1rm_history(values), 'Bench Press (Barbell)')
    assert result['status'] == 'progressing'
    assert result['percent_change'] < 0.02


def test_drop_is_declining_not_plateau():
    df = _e1rm_history([150, 150, 148, 144, 140, 138])
    result = check_plateau(df, 'Bench Press (Barbell)')
    assert result['status'] == 'declining'
    assert result['is_plateau'] is True


def test_untrained_lift_is_inactive():
    # bench stopped in February; squat kept going until June
    bench = _e1rm_history([100, 102, 104, 106, 108, 110])
    squat = _e1rm_history([200] * 26).assign(exercise='Squat (Barbell)')
    df = pd.concat([bench, squat], ignore_index=True)

    assert check_plateau(df, 'Bench Press (Barbell)')['status'] == 'inactive'
    outlook = long_term_outlook(df, 'Bench Press (Barbell)')
    assert 'error' in outlook
    assert outlook['current_smoothed_e1rm_lbs'] == 110


def test_falling_trend_can_project_below_today():
    df = _e1rm_history([160, 157, 154, 151, 148, 145])
    low, high = long_term_outlook(df, 'Bench Press (Barbell)')['outlook_3mo_range_lbs']
    assert low < 145


def test_noisy_falling_trend_never_projects_above_today():
    # noisy decline: the noise spread must not push the band above the current level
    raw = [170, 150, 166, 146, 160, 140, 154, 136]
    rolling = [165, 162, 160, 157, 155, 152, 150, 147]
    df = _e1rm_history(rolling).assign(e1rm=[float(v) for v in raw])
    low, high = long_term_outlook(df, 'Bench Press (Barbell)')['outlook_3mo_range_lbs']
    assert high <= 147 + 1e-6
    assert low < high
