import pandas as pd
import pytest

from predict import check_plateau, long_term_outlook


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


def _two_machine_history():
    """Progressing on machine 0 (~300), then six flat weeks on machine 1 (~150)."""
    first = _e1rm_history([280, 290, 300, 310, 320]).assign(machine=0)
    second = _e1rm_history([150] * 6, start='2026-02-05').assign(machine=1)
    return pd.concat([first, second], ignore_index=True)


def test_plateau_is_judged_on_the_current_machine():
    # against machine 0 this looks like a -50% collapse; on machine 1 it's simply flat
    result = check_plateau(_two_machine_history(), 'Bench Press (Barbell)')
    assert result['status'] == 'plateau'
    assert result['percent_change'] == 0


def test_new_machine_needs_its_own_sessions_for_an_outlook():
    df = _two_machine_history()
    df = df[df['date'] <= pd.Timestamp('2026-02-12')]  # only 2 sessions on machine 1
    outlook = long_term_outlook(df, 'Bench Press (Barbell)')
    assert 'new machine' in outlook['error']
    assert outlook['current_smoothed_e1rm_lbs'] == 150


def test_outlook_is_a_learned_ordered_interval():
    df = _e1rm_history([150 + 2 * i for i in range(12)])
    result = long_term_outlook(df, 'Bench Press (Barbell)')
    low, high = result['outlook_3mo_range_lbs']
    assert result['method'] == 'learned'
    assert low <= result['outlook_3mo_median_lbs'] <= high


def test_interval_scale_widens_without_moving_the_center():
    df = _e1rm_history([150 + 2 * i for i in range(12)])
    base = long_term_outlook(df, 'Bench Press (Barbell)')
    wide = long_term_outlook(df, 'Bench Press (Barbell)', interval_scale=2.0)
    (lo1, hi1), (lo2, hi2) = base['outlook_3mo_range_lbs'], wide['outlook_3mo_range_lbs']
    assert (hi2 - lo2) > (hi1 - lo1) * 1.5
    assert (lo1 + hi1) / 2 == pytest.approx((lo2 + hi2) / 2, rel=0.01)
