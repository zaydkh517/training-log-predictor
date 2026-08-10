import pandas as pd
import pytest

from features import build_e1rm_df, build_modeling_df, build_rep_features, load_and_clean


def test_e1rm_formula():
    modeling_df = pd.DataFrame({
        'date': pd.to_datetime(['2026-01-01']),
        'exercise': ['Bench Press (Barbell)'],
        'weight': [135.0],
        'reps': [8],
    })

    result = build_e1rm_df(modeling_df)

    expected_e1rm = 135 * (1 + 8 / 30)  # Epley formula, worked out by hand: 171.0
    assert result['e1rm'].iloc[0] == round(expected_e1rm, 2)


def test_bodyweight_exercises_excluded():
    combined = pd.DataFrame({
        'date': pd.to_datetime(['2026-01-01', '2026-01-01']),
        'exercise': ['Pull Up', 'Bench Press (Barbell)'],
        'weight': [0.0, 135.0],
        'reps': [10, 8],
        'set_order': [1, 1],
        'source': ['strong', 'strong'],
    })

    result = build_modeling_df(combined)

    assert 'Pull Up' not in result['exercise'].values
    assert 'Bench Press (Barbell)' in result['exercise'].values


def test_prior_set_features_shift_correctly():
    # One earlier session (to establish a prior rolling e1RM), then a
    # three-set session whose prior-set features we check exactly.
    modeling_df = pd.DataFrame({
        'date': pd.to_datetime(['2026-01-01', '2026-01-08', '2026-01-08', '2026-01-08']),
        'exercise': ['Bench Press (Barbell)'] * 4,
        'weight': [100.0, 100.0, 100.0, 95.0],
        'reps': [10, 8, 7, 6],
        'set_order': [1, 1, 2, 3],
        'source': ['strong'] * 4,
    })
    e1rm_df = build_e1rm_df(modeling_df)

    usable = build_rep_features(modeling_df, e1rm_df)

    # Set 1 of each session has no prior set, and the first-ever session has
    # no earlier e1RM -- both get dropped. Only sets 2 and 3 of Jan 8 survive.
    assert len(usable) == 2

    set2 = usable[usable['set_order'] == 2].iloc[0]
    set3 = usable[usable['set_order'] == 3].iloc[0]

    # each set's "prior" must be the previous set of the SAME session
    assert set2['prior_set_reps'] == 8
    assert set2['prior_set_weight'] == 100.0
    assert set3['prior_set_reps'] == 7
    assert set3['prior_set_weight'] == 100.0

    # session_volume_so_far excludes the current set
    assert set2['session_volume_so_far'] == 100.0 * 8
    assert set3['session_volume_so_far'] == 100.0 * 8 + 100.0 * 7

    # rolling_e1rm must come from a STRICTLY earlier date (no same-day leakage):
    # Jan 1's e1RM = 100 * (1 + 10/30) = 133.33
    assert set2['rolling_e1rm'] == 133.33


def test_strong_csv_missing_columns_rejected(tmp_path):
    bad_csv = tmp_path / 'not_strong.csv'
    bad_csv.write_text('foo,bar\n1,2\n')

    with pytest.raises(ValueError, match='missing column'):
        load_and_clean(strong_path=bad_csv)


def test_no_files_rejected():
    with pytest.raises(ValueError):
        load_and_clean()