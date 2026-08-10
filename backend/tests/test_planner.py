from app import find_best_weight, plan_session, PlanSessionRequest


def test_weight_search_stays_near_prior_set():
    # Heavy prior set (225x2 against a 240 e1RM), then a request for 8-12 reps.
    # An unbounded search dives to absurdly light weights the model has no
    # data for; the guardrail must keep it within 20% of the prior weight.
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
    # Moderate scenario: 100x10 against a 150 e1RM. A wide 6-15 rep target is
    # comfortably reachable near the prior weight, so the search should return
    # an in-band weight whose prediction lands inside the target range.
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
    # 225x2 first set, then asking for 8-12: not reachable in-band, so the
    # planned set must carry a note saying the target was missed.
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
