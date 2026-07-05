import pandas as pd

strong_df = pd.read_csv("data/strong_export.csv")
hevy_df = pd.read_csv("data/hevy_export.csv")

# --- Fix date formatting ---
strong_df['date'] = pd.to_datetime(strong_df['Date'], format='%Y-%m-%d %H:%M:%S')
hevy_df['date'] = pd.to_datetime(hevy_df['start_time'], format='%d %b %Y, %H:%M')

# --- Fix mislabeled exercises by me ---
strong_df['Exercise Name'] = strong_df['Exercise Name'].replace('Squat (Band)', 'Squat (Barbell)')

# --- Map Hevy's naming onto Strong's naming for exercises confirmed as the same movement ---
HEVY_TO_CANONICAL = {
    'Chest Fly (Machine)': 'Pec Deck (Machine)',
    'Rear Delt Reverse Fly (Machine)': 'Reverse Fly (Machine)',
    'Triceps Pushdown': 'Triceps Pushdown (Cable - Straight Bar)',
    'Seated Calf Raise': 'Seated Calf Raise (Machine)',
    'Shoulder Press (Dumbbell)': 'Seated Overhead Press (Dumbbell)',
    'Overhead Triceps Extension (Cable)': 'Triceps Extension (Cable)',
}
hevy_df['exercise_title'] = hevy_df['exercise_title'].replace(HEVY_TO_CANONICAL)

# --- Rename each app's columns onto one shared schema ---
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

# --- Stack both into one continuous timeline ---
combined = pd.concat([strong_clean, hevy_clean], ignore_index=True)
combined = combined.sort_values('date').reset_index(drop=True)

print("\n--- Combined dataset ---")
print("Total rows:", len(combined))
print("Date range:", combined['date'].min(), "to", combined['date'].max())
print("Unique exercises:", sorted(combined['exercise'].unique()))
print(combined.head(5))
print(combined.tail(5))

