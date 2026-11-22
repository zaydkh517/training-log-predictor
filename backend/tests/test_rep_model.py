import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app import PlanSessionRequest, plan_session
from rep_model import (
    MIN_PERSONAL_SETS, RepModel, carry_reps, epley_reps, fit_default, fit_personal,
)


def _lifter(n_sessions, drop_per_set, seed=0):
    """Synthetic usable sets: 4 straight sets per session, losing `drop_per_set` reps each set."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sessions):
        e1rm = 200 + i
        weight = 5 * round(e1rm * 0.75 / 5)
        reps = round(epley_reps(weight, e1rm))
        for set_number in range(2, 5):
            new_reps = max(1, reps - drop_per_set + int(rng.integers(-1, 2)))
            rows.append(dict(weight=float(weight), set_number=set_number, prior_set_reps=float(reps),
                             prior_set_weight=float(weight), rolling_e1rm=float(e1rm), reps=float(new_reps)))
            reps = new_reps
    return pd.DataFrame(rows)


def _mae(model, sets):
    pred = model.predict_many(sets['weight'], sets['set_number'], sets['prior_set_reps'],
                              sets['prior_set_weight'], sets['rolling_e1rm'])
    return float(np.mean(np.abs(sets['reps'] - pred)))


def test_carry_keeps_same_day_e1rm():
    # 100 x 10 implies e1RM 133.33; at 110 lb that e1RM allows ~6.4 reps
    assert carry_reps(110, 100, 10) == pytest.approx(133.333 / 110 * 30 - 30, abs=1e-2)
    assert carry_reps(100, 100, 8) == pytest.approx(8)


def test_json_round_trip(tmp_path):
    model = fit_default(_lifter(40, 1))
    model.save(tmp_path / 'm.json')
    loaded = RepModel.load(tmp_path / 'm.json')
    assert loaded.predict(150, 3, 8, 150, 200) == pytest.approx(model.predict(150, 3, 8, 150, 200))


def test_mismatched_model_rejected():
    d = fit_default(_lifter(40, 1)).to_dict()
    with pytest.raises(ValueError):
        RepModel.from_dict({**d, 'features': ['weight'] * 5})
    with pytest.raises(ValueError):
        RepModel.from_dict({**d, 'coef': d['coef'][:3]})


def test_prediction_is_physically_possible():
    model = fit_default(_lifter(40, 1))
    for weight in (150, 190, 199):
        reps = model.predict(weight, 3, 12, 150, 200)
        assert 0 <= reps <= epley_reps(weight, 200) + 1e-9


def test_short_history_keeps_default():
    default = fit_default(_lifter(40, 1))
    tiny = _lifter(3, 3)
    assert len(tiny) < MIN_PERSONAL_SETS
    assert fit_personal(tiny, default) is default


def test_personal_fit_learns_this_lifters_fatigue():
    # default learned a 1-rep drop per set; this lifter drops 3 -- the personal fit should notice
    default = fit_default(_lifter(60, 1, seed=1))
    train, test = _lifter(60, 3, seed=2), _lifter(20, 3, seed=3)
    personal = fit_personal(train, default)
    assert personal.source == 'personal'
    assert _mae(personal, test) < _mae(default, test)


def test_personal_fit_shrinks_toward_default():
    # more data -> further from the default, never a jump on a handful of sets
    default = fit_default(_lifter(60, 1, seed=1))
    lifter = _lifter(80, 3, seed=2)
    small = fit_personal(lifter.head(MIN_PERSONAL_SETS), default)
    large = fit_personal(lifter, default)
    gap = lambda m: abs(m.intercept - default.intercept)
    assert 0 < gap(small) < gap(large)


def test_planner_uses_supplied_personal_model():
    default = fit_default(_lifter(60, 1, seed=1))
    personal = fit_personal(_lifter(80, 3, seed=2), default)
    request = PlanSessionRequest(
        exercise='Bench Press (Barbell)', rolling_e1rm=200.0, total_sets=3,
        first_set_weight=150.0, target_min_reps=6, target_max_reps=10,
        rep_model=personal.to_dict(),
    )
    assert plan_session(request)['rep_model_source'] == 'personal'


def test_planner_rejects_malformed_model():
    bad = {**fit_default(_lifter(40, 1)).to_dict(), 'scale': [0, 1, 1, 1, 1]}
    with pytest.raises(ValidationError):
        PlanSessionRequest(
            exercise='Bench Press (Barbell)', rolling_e1rm=200.0, total_sets=3,
            first_set_weight=150.0, target_min_reps=6, target_max_reps=10, rep_model=bad,
        )


def test_mislogged_weight_jumps_dont_distort_the_fit():
    # two variants logged under one name: 245 x8 then 85 x7 implies ~80 reps under the
    # baseline. A few rows like that used to wreck the fit; they're outside the planner's
    # range, so they must be ignored.
    clean = _lifter(60, 1, seed=1)
    junk = pd.DataFrame([dict(weight=85.0, set_number=2, prior_set_reps=8.0, prior_set_weight=245.0,
                              rolling_e1rm=310.0, reps=7.0)] * 10)
    test = _lifter(20, 1, seed=4)
    assert _mae(fit_default(pd.concat([clean, junk])), test) == pytest.approx(_mae(fit_default(clean), test))
