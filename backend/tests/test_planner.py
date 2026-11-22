import pytest
from pydantic import ValidationError

from app import find_best_weight, plan_session, PlanSessionRequest, MAX_PLANNED_SETS


def test_weight_search_stays_near_prior_set():
    # heavy prior set: an unbounded search would dive to weights the model has no data for
    prior_weight = 225.0
    weight, reps = find_best_weight(
        prior_weight=prior_weight,
        prior_reps=2,
        rolling_e1rm=240.0,
        set_number=2,
        target_min=8,
        target_max=12,
    )

    assert weight >= prior_weight * 0.8   # never drop more than 20%
    assert weight <= prior_weight * 1.2   # never jump more than 20%


def test_reachable_target_is_hit_in_band():
    # moderate set with a wide target: reachable near the prior weight
    prior_weight = 100.0
    weight, reps = find_best_weight(
        prior_weight=prior_weight,
        prior_reps=10,
        rolling_e1rm=150.0,
        set_number=2,
        target_min=6,
        target_max=15,
    )

    assert prior_weight * 0.8 <= weight <= prior_weight * 1.2
    assert 6 <= reps <= 15


def test_weights_are_gym_loadable():
    # light lift where the 20% cap makes the search start off the 5 lb lattice
    weight, reps = find_best_weight(
        prior_weight=30.0,
        prior_reps=12,
        rolling_e1rm=45.0,
        set_number=2,
        target_min=8,
        target_max=12,
    )

    assert weight % 5 == 0


def test_unreachable_target_gets_note():
    # 8-12 reps isn't reachable within 20% of a near-max set -> note required
    request = PlanSessionRequest(
        exercise='Bench Press (Barbell)', rolling_e1rm=240.0, total_sets=2,
        first_set_weight=225.0, target_min_reps=8, target_max_reps=12,
    )
    result = plan_session(request)

    set2 = result['sets'][1]
    assert set2['predicted_reps'] < 8
    assert 'note' in set2


def test_reachable_target_has_no_note():
    request = PlanSessionRequest(
        exercise='Bench Press (Barbell)', rolling_e1rm=150.0, total_sets=2,
        first_set_weight=100.0, target_min_reps=6, target_max_reps=15,
    )
    result = plan_session(request)

    set2 = result['sets'][1]
    assert 6 <= set2['predicted_reps'] <= 15
    assert 'note' not in set2


def test_plan_never_exceeds_estimated_max():
    # regression: the uncapped model ramped 230 -> 275 lb against a 240 lb e1RM
    request = PlanSessionRequest(
        exercise='Bench Press (Barbell)', rolling_e1rm=240.0, total_sets=6,
        first_set_weight=230.0, target_min_reps=1, target_max_reps=3,
    )
    result = plan_session(request)

    for s in result['sets']:
        if s.get('weight') is not None:
            assert s['weight'] <= 240.0
            fresh_reps = 30 * (240.0 / s['weight'] - 1)
            assert s['predicted_reps'] <= round(fresh_reps)


def test_search_can_climb_past_ten_pounds():
    # 15 reps at 200 with a 6-8 target: the right next set is well over +10 lb
    weight, reps = find_best_weight(
        prior_weight=200.0, prior_reps=15, rolling_e1rm=300.0,
        set_number=2, target_min=6, target_max=8,
    )
    assert weight > 210.0


def test_first_set_outside_target_gets_note():
    # 100 lb at a 200 lb e1RM is ~30 reps, nowhere near 5-8
    request = PlanSessionRequest(
        exercise='Bench Press (Barbell)', rolling_e1rm=200.0, total_sets=1,
        first_set_weight=100.0, target_min_reps=5, target_max_reps=8,
    )
    set1 = plan_session(request)['sets'][0]
    assert 'note' in set1
    assert 'lb' in set1['note']


def test_first_set_in_target_has_no_note():
    request = PlanSessionRequest(
        exercise='Bench Press (Barbell)', rolling_e1rm=200.0, total_sets=1,
        first_set_weight=160.0, target_min_reps=5, target_max_reps=8,
    )
    assert 'note' not in plan_session(request)['sets'][0]


def test_set_count_is_bounded():
    with pytest.raises(ValidationError):
        PlanSessionRequest(
            exercise='Bench Press (Barbell)', rolling_e1rm=200.0, total_sets=MAX_PLANNED_SETS + 1,
            first_set_weight=150.0, target_min_reps=5, target_max_reps=8,
        )
