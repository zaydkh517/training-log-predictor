import pandas as pd

from predict import (
    check_plateau,
    long_term_outlook,
    MIN_GROWTH_PCT,
    MAX_GROWTH_PCT,
    EXPERIENCE_K,
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

    result = long_term_outlook(df, 'Bench Press (Barbell)')

    # recompute the ceiling exactly as predict.py defines it
    n_sessions = len(rolling)
    growth_cap = MIN_GROWTH_PCT + (MAX_GROWTH_PCT - MIN_GROWTH_PCT) * EXPERIENCE_K / (n_sessions + EXPERIENCE_K)
    ceiling = 150 * (1 + growth_cap)

    low, high = result['outlook_6mo_range_lbs']
    assert high <= ceiling + 1e-6
    assert low >= 150  # never projected below current level