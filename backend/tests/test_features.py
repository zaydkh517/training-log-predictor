import pandas as pd
import pytest

from features import KG_TO_LB, build_dataset, build_e1rm_df, build_modeling_df, build_rep_features, load_and_clean


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
    # one earlier session to establish a rolling e1RM, then a 3-set session
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

    # only sets 2 and 3 of Jan 8 have both a prior set and an earlier e1RM
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

    # no same-day leakage: rolling e1RM is Jan 1's alone, 100 * (1 + 10/30)
    assert set2['rolling_e1rm'] == 133.33


def test_strong_csv_missing_columns_rejected(tmp_path):
    bad_csv = tmp_path / 'not_strong.csv'
    bad_csv.write_text('foo,bar\n1,2\n')

    with pytest.raises(ValueError, match='missing column'):
        load_and_clean(strong_path=bad_csv)


def test_no_files_rejected():
    with pytest.raises(ValueError):
        load_and_clean()

def _sets(rows):
    """rows: (date, exercise, weight, reps); set_order counts up within each session."""
    df = pd.DataFrame(rows, columns=['date', 'exercise', 'weight', 'reps'])
    df['date'] = pd.to_datetime(df['date'])
    df['set_order'] = df.groupby(['date', 'exercise']).cumcount() + 1
    df['source'] = 'strong'
    return df


def test_best_set_is_highest_e1rm_not_heaviest():
    # 225x2 (e1RM 240) is heavier, but 205x8 (e1RM 259.67) is the stronger set
    df = _sets([('2026-01-01', 'Bench Press (Barbell)', 225.0, 2),
                ('2026-01-01', 'Bench Press (Barbell)', 205.0, 8)])
    result = build_e1rm_df(df)
    assert result['e1rm'].iloc[0] == round(205 * (1 + 8 / 30), 2)


def test_high_rep_set_only_counts_when_alone():
    # 100x20 has a bigger Epley number than 135x5, but Epley isn't trustworthy at 20 reps
    mixed = _sets([('2026-01-01', 'Bench Press (Barbell)', 135.0, 5),
                   ('2026-01-01', 'Bench Press (Barbell)', 100.0, 20)])
    assert build_e1rm_df(mixed)['reps'].iloc[0] == 5

    only_high = _sets([('2026-01-01', 'Bench Press (Barbell)', 100.0, 20)])
    assert build_e1rm_df(only_high)['reps'].iloc[0] == 20


def test_cardio_and_blank_rows_dropped():
    df = _sets([('2026-01-01', 'Running', 0.0, 0),
                ('2026-01-01', 'Treadmill', float('nan'), float('nan')),
                ('2026-01-01', 'Bench Press (Barbell)', 135.0, 8)])
    result = build_modeling_df(df)
    assert list(result['exercise']) == ['Bench Press (Barbell)']


def test_only_cardio_is_a_clear_error(tmp_path):
    csv = tmp_path / 'strong.csv'
    csv.write_text(
        'Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE\n'
        '2026-01-01 10:00:00,Run,30m,Running,1,0,0,3,1800,\n'
    )
    with pytest.raises(ValueError, match='No weighted sets'):
        build_dataset(strong_path=csv)


HEVY_HEADER = ('"title","start_time","end_time","description","exercise_title","superset_id",'
               '"exercise_notes","set_index","set_type","{weight_col}","reps","distance_miles",'
               '"duration_seconds","rpe"\n')


def _hevy_row(set_type, weight, reps):
    return (f'"Push","1 Jan 2026, 10:00","1 Jan 2026, 11:00","","Bench Press (Barbell)",,"",0,'
            f'"{set_type}",{weight},{reps},,,\n')


def test_hevy_kg_export_converted_to_lb(tmp_path):
    csv = tmp_path / 'hevy.csv'
    csv.write_text(HEVY_HEADER.format(weight_col='weight_kg') + _hevy_row('normal', 100, 5))
    result = load_and_clean(hevy_path=csv)
    assert result['weight'].iloc[0] == pytest.approx(100 * KG_TO_LB)


def test_hevy_warmups_excluded(tmp_path):
    csv = tmp_path / 'hevy.csv'
    csv.write_text(HEVY_HEADER.format(weight_col='weight_lbs')
                   + _hevy_row('warmup', 95, 10) + _hevy_row('normal', 185, 5))
    result = load_and_clean(hevy_path=csv)
    assert list(result['weight']) == [185]
