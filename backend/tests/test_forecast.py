import numpy as np
import pandas as pd

from forecast import backtest, features_at, interval_scale, load_model


def _sessions(values, start='2026-01-01', freq='7D', machine=0, exercise='Bench Press (Barbell)'):
    dates = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({'date': dates, 'exercise': exercise, 'machine': machine,
                         'e1rm': [float(v) for v in values], 'rolling_e1rm': [float(v) for v in values]})


def test_recent_rate_uses_the_session_nearest_90_days_back():
    s = _sessions([100 + i for i in range(20)])          # weekly, +1 lb a week
    dates, rolling = s['date'].to_numpy(), s['rolling_e1rm'].to_numpy()
    f = features_at(dates, rolling, 19, dates[0])
    assert f['prev_gap'] == 91                           # 13 weeks back is nearest to 90 days
    assert f['prev_rate'] == np.log(119 / 106) * 30 / 91
    assert f['train_age'] == 19 * 7


def test_no_recent_rate_without_a_session_30_days_back():
    s = _sessions([100, 101, 102], freq='5D')
    f = features_at(s['date'].to_numpy(), s['rolling_e1rm'].to_numpy(), 2, s['date'].to_numpy()[0])
    assert np.isnan(f['prev_rate']) and np.isnan(f['prev_gap'])


def test_backtest_only_uses_resolved_outcomes_on_the_same_machine():
    a = _sessions([100 + i for i in range(30)])                         # 30 weeks on machine 0
    b = _sessions([50] * 30, start='2026-01-02', machine=1)             # interleaved machine 1
    past = backtest(pd.concat([a, b], ignore_index=True))
    # every outcome is a same-machine session 60-200 days after a session with 4+ before it
    assert past['horizon'].between(60, 200).all()
    assert np.isclose(past['y'], 0).sum() == len(past) / 2              # machine 1 is flat, never mixed with 0
    assert len(past) == 2 * (30 - 4 - 9)                                # last 9 weeks have no outcome yet


def test_interval_scale_is_neutral_without_history_and_never_narrows():
    model = load_model()
    assert interval_scale(model, _sessions([100, 101, 102])) == (1.0, 0)
    scale, n = interval_scale(model, _sessions([100 + 3 * i for i in range(40)]))
    assert scale >= 1.0 and n > 0
