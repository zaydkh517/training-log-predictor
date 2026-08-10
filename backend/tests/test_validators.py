import pytest
from pydantic import ValidationError

from app import PredictSetRequest, PlanSessionRequest


def test_set_one_with_no_prior_is_valid():
    req = PredictSetRequest(exercise='Bench', weight=100, rolling_e1rm=150, set_number=1)
    assert req.prior_set_reps is None


def test_set_one_with_prior_rejected():
    with pytest.raises(ValidationError):
        PredictSetRequest(exercise='Bench', weight=100, rolling_e1rm=150,
                          set_number=1, prior_set_reps=8, prior_set_weight=100)


def test_later_set_without_prior_rejected():
    with pytest.raises(ValidationError):
        PredictSetRequest(exercise='Bench', weight=100, rolling_e1rm=150, set_number=2)


def test_half_a_prior_rejected():
    with pytest.raises(ValidationError):
        PredictSetRequest(exercise='Bench', weight=100, rolling_e1rm=150,
                          set_number=2, prior_set_reps=8)


def test_inverted_rep_range_rejected():
    with pytest.raises(ValidationError):
        PlanSessionRequest(exercise='Bench', rolling_e1rm=150, total_sets=3,
                           first_set_weight=100, target_min_reps=10, target_max_reps=5)