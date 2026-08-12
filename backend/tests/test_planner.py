from app import find_best_weight, plan_session, PlanSessionRequest


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
    assert weight <= prior_weight + 10    # never jump past the +10 start


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

    assert prior_weight * 0.8 <= weight <= prior_weight + 10
    assert 6 <= reps <= 15


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
