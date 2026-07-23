import pandas as pd

BODYWEIGHT_EXERCISES = [
    'Pull Up',
    'Chin Up',
    'Push Up',
    'Dip',
    'Muscle Up',
    'Sit Up',
    'Crunch',
    'Plank',
    'Burpee',
    'Mountain Climber',
    'Air Squat',
    'Lunge',
    'Leg Raise Parallel Bars',
    'Hanging Leg Raise',
    'Toes To Bar',
    'Pistol Squat',
    'Glute Bridge',
    'Inverted Row',
]

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


def load_and_clean(strong_path=None, hevy_path=None):
    # Either path can be omitted -- most real users only export from one app, not both.
    pieces = []

    if strong_path is not None:
        strong_df = pd.read_csv(strong_path)
        strong_df['date'] = pd.to_datetime(strong_df['Date'], format='%Y-%m-%d %H:%M:%S')
        strong_df['Exercise Name'] = strong_df['Exercise Name'].replace('Squat (Band)', 'Squat (Barbell)')

        strong_clean = strong_df.rename(columns={
            'Exercise Name': 'exercise',
            'Weight': 'weight',
            'Reps': 'reps',
            'Set Order': 'set_order',
        })[['date', 'exercise', 'weight', 'reps', 'set_order']]
        strong_clean['source'] = 'strong'
        pieces.append(strong_clean)

    if hevy_path is not None:
        hevy_df = pd.read_csv(hevy_path)
        hevy_df['date'] = pd.to_datetime(hevy_df['start_time'], format='%d %b %Y, %H:%M')
        hevy_df['exercise_title'] = hevy_df['exercise_title'].replace(HEVY_TO_CANONICAL)

        hevy_clean = hevy_df.rename(columns={
            'exercise_title': 'exercise',
            'weight_lbs': 'weight',
            'reps': 'reps',
            'set_index': 'set_order',
        })[['date', 'exercise', 'weight', 'reps', 'set_order']]
        hevy_clean['source'] = 'hevy'
        pieces.append(hevy_clean)

    if not pieces:
        raise ValueError("load_and_clean needs at least one of strong_path or hevy_path")

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)
    return combined


#get rid of bodyweight exercises
def build_modeling_df(combined):
    return combined[~combined['exercise'].isin(BODYWEIGHT_EXERCISES)].copy()

#dataframe of e1rm
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

    e1rm_lookup = e1rm_df[['exercise', 'date', 'rolling_e1rm']].sort_values('date')

    # by='exercise' does the per-exercise grouping merge_asof itself -- no manual loop needed.
    # Both sides must still be sorted by 'date' overall (not by exercise+date) for this to be valid.
    df = pd.merge_asof(
        df.sort_values('date'), e1rm_lookup,
        on='date', by='exercise', direction='backward', allow_exact_matches=False,
    )
    df = df.sort_values(['exercise', 'date', 'set_order']).reset_index(drop=True)

    df['weight_pct_e1rm'] = df['weight'] / df['rolling_e1rm']
    df['formula_pred_reps'] = 30 * (df['rolling_e1rm'] / df['weight'] - 1)

    # drop rows that can't actually be used for training or prediction: no prior
    # set (first set of a session) or no prior e1RM (first-ever session logged
    # for that exercise) -- plus basic sanity filtering on reps.
    usable = df.dropna(
        subset=['prior_set_reps', 'prior_set_weight', 'rolling_e1rm', 'formula_pred_reps']
    ).copy()
    usable = usable[(usable['reps'] > 0) & (usable['reps'] < 50)]
    return usable


def build_dataset(strong_path=None, hevy_path=None):

    combined = load_and_clean(strong_path, hevy_path)
    modeling_df = build_modeling_df(combined)
    e1rm_df = build_e1rm_df(modeling_df)
    usable_sets = build_rep_features(modeling_df, e1rm_df)
    return {
        'combined': combined,
        'modeling_df': modeling_df,
        'e1rm_df': e1rm_df,
        'usable_sets': usable_sets,
    }
