import math
from statistics import median

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

# map Hevy names onto Strong's so shared exercises line up
HEVY_TO_CANONICAL = {
    'Chest Fly (Machine)': 'Pec Deck (Machine)',
    'Rear Delt Reverse Fly (Machine)': 'Reverse Fly (Machine)',
    'Triceps Pushdown': 'Triceps Pushdown (Cable - Straight Bar)',
    'Seated Calf Raise': 'Seated Calf Raise (Machine)',
    'Shoulder Press (Dumbbell)': 'Seated Overhead Press (Dumbbell)',
    'Overhead Triceps Extension (Cable)': 'Triceps Extension (Cable)',
    'Hanging Leg Raise': 'Leg Raise Parallel Bars',
}


KG_TO_LB = 2.20462
MAX_RELIABLE_REPS = 12

# A session whose e1RM is under 60% (or over ~167%) of a machine's recent level can't be the
# same machine: real strength never moves that much between sessions, but a different cable
# stack or leg-extension machine easily does. Pounds on different machines aren't comparable.
MACHINE_SWITCH_RATIO = 0.6
MACHINE_LEVEL_SESSIONS = 3   # a machine's "recent level" is the median of its last 3 sessions



def load_and_clean(strong_path=None, hevy_path=None):
    # either path can be omitted -- most users only export from one app
    pieces = []

    if strong_path is not None:
        try:
            strong_df = pd.read_csv(strong_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            raise ValueError("That Strong file couldn't be read as a CSV -- check it's the right file and isn't empty")

        required = {'Date', 'Exercise Name', 'Weight', 'Reps', 'Set Order'}
        missing = required - set(strong_df.columns)
        if missing:
            raise ValueError(
                f"This doesn't look like a Strong export -- missing column(s): {', '.join(sorted(missing))}"
            )

        strong_df['date'] = pd.to_datetime(strong_df['Date'], format='%Y-%m-%d %H:%M:%S')
        strong_df['Exercise Name'] = strong_df['Exercise Name'].replace('Squat (Band)', 'Squat (Barbell)')

        # some Strong exports carry a per-row unit; without one, Weight is assumed to be lb
        if 'Weight Unit' in strong_df.columns:
            is_kg = strong_df['Weight Unit'].astype(str).str.strip().str.lower().isin(['kg', 'kgs'])
            strong_df.loc[is_kg, 'Weight'] = pd.to_numeric(strong_df.loc[is_kg, 'Weight'], errors='coerce') * KG_TO_LB

        strong_clean = strong_df.rename(columns={
            'Exercise Name': 'exercise',
            'Weight': 'weight',
            'Reps': 'reps',
            'Set Order': 'set_order',
        })[['date', 'exercise', 'weight', 'reps', 'set_order']]
        strong_clean['source'] = 'strong'
        pieces.append(strong_clean)

    if hevy_path is not None:
        try:
            hevy_df = pd.read_csv(hevy_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            raise ValueError("That Hevy file couldn't be read as a CSV -- check it's the right file and isn't empty")

        # Hevy names the weight column after the user's unit setting
        weight_col = next((c for c in ('weight_lbs', 'weight_kg') if c in hevy_df.columns), None)
        required = {'start_time', 'exercise_title', 'reps', 'set_index'}
        missing = required - set(hevy_df.columns)
        if weight_col is None:
            missing.add('weight_lbs or weight_kg')
        if missing:
            raise ValueError(
                f"This doesn't look like a Hevy export -- missing column(s): {', '.join(sorted(missing))}"
            )

        hevy_df['date'] = pd.to_datetime(hevy_df['start_time'], format='%d %b %Y, %H:%M')
        hevy_df['exercise_title'] = hevy_df['exercise_title'].replace(HEVY_TO_CANONICAL)
        hevy_df['weight'] = pd.to_numeric(hevy_df[weight_col], errors='coerce')
        if weight_col == 'weight_kg':
            hevy_df['weight'] *= KG_TO_LB

        # warm-ups aren't working sets: they'd never be the best set, but they'd pollute prior-set features
        if 'set_type' in hevy_df.columns:
            hevy_df = hevy_df[hevy_df['set_type'].astype(str).str.lower() != 'warmup']

        hevy_clean = hevy_df.rename(columns={
            'exercise_title': 'exercise',
            'set_index': 'set_order',
        })[['date', 'exercise', 'weight', 'reps', 'set_order']]
        hevy_clean['source'] = 'hevy'
        pieces.append(hevy_clean)

    if not pieces:
        raise ValueError("load_and_clean needs at least one of strong_path or hevy_path")

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)
    return combined


def build_modeling_df(combined):
    """Loaded strength sets only: drops bodyweight moves and rows with no usable weight x reps
    (cardio, timed holds, blank sets), which would otherwise turn into NaN or zero e1RMs."""
    df = combined[~combined['exercise'].isin(BODYWEIGHT_EXERCISES)].copy()
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['reps'] = pd.to_numeric(df['reps'], errors='coerce')
    df['set_order'] = pd.to_numeric(df['set_order'], errors='coerce')
    return df[(df['weight'] > 0) & (df['reps'] > 0)].copy()


def assign_machines(e1rms):
    """Machine id (0, 1, 2...) for each session of one lift, in date order.

    Each session joins the machine whose recent level it's consistent with -- preferring
    the one used last time -- or starts a new machine if it matches none. Going back to an
    earlier gym picks that machine's history back up instead of starting over.
    """
    recent = []      # per machine: its session e1RMs so far
    ids = []
    current = None

    def distance(machine, value):
        """How far `value` is from this machine's recent level (log ratio), or None if
        it's too far to be the same machine."""
        ratio = value / median(recent[machine][-MACHINE_LEVEL_SESSIONS:])
        if MACHINE_SWITCH_RATIO <= ratio <= 1 / MACHINE_SWITCH_RATIO:
            return abs(math.log(ratio))
        return None

    for value in e1rms:
        if current is not None and distance(current, value) is not None:
            machine = current
        else:
            fits = [(distance(m, value), m) for m in range(len(recent))]
            fits = [(d, m) for d, m in fits if d is not None]
            if fits:
                machine = min(fits)[1]
            else:
                recent.append([])
                machine = len(recent) - 1
        recent[machine].append(value)
        ids.append(machine)
        current = machine
    return ids


def build_e1rm_df(modeling_df):
    """One e1RM per exercise per session -- the set with the highest Epley estimate --
    tagged with the machine it was done on, plus a 28-day rolling average per machine."""
    df = modeling_df.copy()
    df['e1rm'] = (df['weight'] * (1 + df['reps'] / 30)).round(2)

    # Epley overestimates past ~12 reps, so a high-rep set only counts when it's all the session has
    df['reliable'] = df['reps'] <= MAX_RELIABLE_REPS
    df = df.sort_values(['date', 'exercise', 'reliable', 'e1rm'])
    e1rm_df = df.drop_duplicates(subset=['date', 'exercise'], keep='last').drop(columns='reliable')
    e1rm_df = e1rm_df.sort_values(['exercise', 'date']).reset_index(drop=True)
    e1rm_df['machine'] = 0
    e1rm_df['rolling_e1rm'] = 0.0
    for _, lift in e1rm_df.groupby('exercise'):
        e1rm_df.loc[lift.index, 'machine'] = assign_machines(lift['e1rm'].tolist())

    # smooth within a machine only, so a switch doesn't drag the average through the gap
    for _, sessions in e1rm_df.groupby(['exercise', 'machine']):
        e1rm_df.loc[sessions.index, 'rolling_e1rm'] = (
            sessions.rolling('28D', on='date')['e1rm'].mean().to_numpy()
        )
    return e1rm_df


def build_rep_features(modeling_df, e1rm_df):
    df = modeling_df.dropna(subset=['weight', 'reps']).copy()
    df = df.sort_values(['exercise', 'date', 'set_order']).reset_index(drop=True)

    df['prior_set_reps'] = df.groupby(['exercise', 'date'])['reps'].shift(1)
    df['prior_set_weight'] = df.groupby(['exercise', 'date'])['weight'].shift(1)
    df['set_number'] = df.groupby(['exercise', 'date']).cumcount() + 1
    df['set_volume'] = df['weight'] * df['reps']
    df['session_volume_so_far'] = df.groupby(['exercise', 'date'])['set_volume'].cumsum() - df['set_volume']

    # each set takes its session's machine, and its rolling e1RM from earlier sessions on that
    # same machine (a different machine's pounds would make %-of-e1RM meaningless)
    df = df.merge(e1rm_df[['exercise', 'date', 'machine']], on=['exercise', 'date'], how='left')
    e1rm_lookup = e1rm_df[['exercise', 'machine', 'date', 'rolling_e1rm']].sort_values('date')

    # merge_asof groups by exercise + machine; both sides must be sorted by date overall
    df = pd.merge_asof(
        df.sort_values('date'), e1rm_lookup,
        on='date', by=['exercise', 'machine'], direction='backward', allow_exact_matches=False,
    )
    df = df.sort_values(['exercise', 'date', 'set_order']).reset_index(drop=True)

    df['weight_pct_e1rm'] = df['weight'] / df['rolling_e1rm']
    df['formula_pred_reps'] = 30 * (df['rolling_e1rm'] / df['weight'] - 1)

    # drop rows unusable for training: no prior set, no prior e1RM, or nonsense rep counts
    usable = df.dropna(
        subset=['prior_set_reps', 'prior_set_weight', 'rolling_e1rm', 'formula_pred_reps']
    ).copy()
    usable = usable[(usable['reps'] > 0) & (usable['reps'] < 50)]
    return usable


def build_dataset(strong_path=None, hevy_path=None):
    combined = load_and_clean(strong_path, hevy_path)
    modeling_df = build_modeling_df(combined)
    if modeling_df.empty:
        raise ValueError("No weighted sets found -- the file only has bodyweight, cardio or empty sets")
    e1rm_df = build_e1rm_df(modeling_df)
    usable_sets = build_rep_features(modeling_df, e1rm_df)
    return {
        'combined': combined,
        'modeling_df': modeling_df,
        'e1rm_df': e1rm_df,
        'usable_sets': usable_sets,
    }
