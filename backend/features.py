import pandas as pd

BODYWEIGHT_EXERCISES = ['Pull Up', 'Leg Raise Parallel Bars']

# Hevy and Strong name some of the same exercises differently. Map Hevy's
# names onto Strong's so they're treated as the same exercise.
HEVY_TO_CANONICAL = {
    'Chest Fly (Machine)': 'Pec Deck (Machine)',
    'Rear Delt Reverse Fly (Machine)': 'Reverse Fly (Machine)',
    'Triceps Pushdown': 'Triceps Pushdown (Cable - Straight Bar)',
    'Seated Calf Raise': 'Seated Calf Raise (Machine)',
    'Shoulder Press (Dumbbell)': 'Seated Overhead Press (Dumbbell)',
    'Overhead Triceps Extension (Cable)': 'Triceps Extension (Cable)',
    'Hanging Leg Raise': 'Leg Raise Parallel Bars',
}


def load_and_clean(strong_path, hevy_path):
    
    strong_df = pd.read_csv(strong_path)
    hevy_df = pd.read_csv(hevy_path)

    strong_df['date'] = pd.to_datetime(strong_df['Date'], format='%Y-%m-%d %H:%M:%S')
    hevy_df['date'] = pd.to_datetime(hevy_df['start_time'], format='%d %b %Y, %H:%M')

    strong_df['Exercise Name'] = strong_df['Exercise Name'].replace('Squat (Band)', 'Squat (Barbell)')
    hevy_df['exercise_title'] = hevy_df['exercise_title'].replace(HEVY_TO_CANONICAL)

    strong_clean = strong_df.rename(columns={
        'Exercise Name': 'exercise',
        'Weight': 'weight',
        'Reps': 'reps',
        'Set Order': 'set_order',
    })[['date', 'exercise', 'weight', 'reps', 'set_order']]
    strong_clean['source'] = 'strong'

    hevy_clean = hevy_df.rename(columns={
        'exercise_title': 'exercise',
        'weight_lbs': 'weight',
        'reps': 'reps',
        'set_index': 'set_order',
    })[['date', 'exercise', 'weight', 'reps', 'set_order']]
    hevy_clean['source'] = 'hevy'

    combined = pd.concat([strong_clean, hevy_clean], ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)
    return combined


def build_modeling_df(combined):
    return combined[~combined['exercise'].isin(BODYWEIGHT_EXERCISES)].copy()


def build_e1rm_df(modeling_df):
    
    sorted_df = modeling_df.sort_values(['date', 'exercise', 'weight', 'reps'])
    e1rm_df = sorted_df.drop_duplicates(subset=['date', 'exercise'], keep='last').copy()

    e1rm_df['e1rm'] = (e1rm_df['weight'] * (1 + e1rm_df['reps'] / 30)).round(2)

    rolling = (
        e1rm_df.groupby('exercise')
        .rolling('28D', on='date')['e1rm']
        .mean()
        .reset_index()
        .rename(columns={'e1rm': 'rolling_e1rm'})
    )
    e1rm_df = e1rm_df.merge(rolling[['exercise', 'date', 'rolling_e1rm']], on=['exercise', 'date'], how='left')

    e1rm_df['intensity_trend'] = (e1rm_df['weight'] / e1rm_df['rolling_e1rm']).round(4)
    return e1rm_df


def build_rep_features(modeling_df, e1rm_df):
    
    df = modeling_df.dropna(subset=['weight', 'reps']).copy()
    df = df.sort_values(['exercise', 'date', 'set_order']).reset_index(drop=True)

    df['prior_set_reps'] = df.groupby(['exercise', 'date'])['reps'].shift(1)
    df['prior_set_weight'] = df.groupby(['exercise', 'date'])['weight'].shift(1)
    df['set_number'] = df.groupby(['exercise', 'date']).cumcount() + 1

    e1rm_lookup = e1rm_df[['exercise', 'date', 'rolling_e1rm']].sort_values(['exercise', 'date'])

    parts = []
    for ex, grp in df.groupby('exercise'):
        lk = e1rm_lookup[e1rm_lookup['exercise'] == ex].sort_values('date')
        grp = grp.sort_values('date')
        merged = pd.merge_asof(
            grp, lk[['date', 'rolling_e1rm']],
            on='date', direction='backward', allow_exact_matches=False,
        )
        parts.append(merged)
    df = pd.concat(parts, ignore_index=True)

    df['weight_pct_e1rm'] = df['weight'] / df['rolling_e1rm']
    df['formula_pred_reps'] = 30 * (df['rolling_e1rm'] / df['weight'] - 1)
    return df


def build_usable_sets(rep_features_df):
    
    usable = rep_features_df.dropna(
        subset=['prior_set_reps', 'prior_set_weight', 'rolling_e1rm', 'formula_pred_reps']
    ).copy()
    usable = usable[(usable['reps'] > 0) & (usable['reps'] < 50)]
    return usable


def build_dataset(strong_path, hevy_path):
    
    combined = load_and_clean(strong_path, hevy_path)
    modeling_df = build_modeling_df(combined)
    e1rm_df = build_e1rm_df(modeling_df)
    rep_features_df = build_rep_features(modeling_df, e1rm_df)
    usable_sets = build_usable_sets(rep_features_df)
    return {
        'combined': combined,
        'modeling_df': modeling_df,
        'e1rm_df': e1rm_df,
        'rep_features_df': rep_features_df,
        'usable_sets': usable_sets,
    }
