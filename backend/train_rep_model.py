"""Train and evaluate the set-level rep model; writes rep_model.json for the API.

Training data (all gitignored):
    data/strong_export.csv, data/hevy_export.csv     -- the original lifter
    data/lifters/*.csv                               -- optional extra lifters, one export
                                                        file per person (Strong or Hevy,
                                                        detected from the header)

Every predictor is scored on the same held-out sets, two ways:
  - session-grouped 5-fold CV: whole workouts held out (sets in one workout share fatigue)
  - time split: train on the earliest 70% of each lifter's history, test on the rest --
    the honest check for a model that will be used on *future* sessions
With 2+ lifters it also runs leave-one-lifter-out: the check for a brand-new user.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_dataset
from rep_model import MAX_WEIGHT_CHANGE, carry_reps, fit_default, in_planner_range

DATA = Path('data')
# the pre-rework model, kept as a comparison point: absolute weights, no baseline
LEGACY_FEATURES = ['weight', 'set_number', 'prior_set_reps', 'prior_set_weight', 'rolling_e1rm', 'weight_pct_e1rm']


def export_kind(path):
    """'strong' or 'hevy', read from the CSV header so filenames don't matter."""
    header = path.open().readline()
    if 'Exercise Name' in header:
        return 'strong'
    if 'exercise_title' in header:
        return 'hevy'
    raise SystemExit(f'{path} is neither a Strong nor a Hevy export')


def load_lifters():
    lifters = {}
    strong, hevy = DATA / 'strong_export.csv', DATA / 'hevy_export.csv'
    if strong.exists() or hevy.exists():
        lifters['lifter_0'] = (strong if strong.exists() else None, hevy if hevy.exists() else None)
    for path in sorted((DATA / 'lifters').glob('*.csv')):
        kind = export_kind(path)
        lifters[f'lifter_{len(lifters)}'] = (path if kind == 'strong' else None, path if kind == 'hevy' else None)

    frames = []
    for name, (strong, hevy) in lifters.items():
        sets = build_dataset(strong_path=strong, hevy_path=hevy)['usable_sets'].copy()
        sets['lifter'] = name
        frames.append(sets)
    if not frames:
        raise SystemExit('no training data found under data/')
    sets = pd.concat(frames, ignore_index=True)
    sets['session'] = sets['lifter'] + '|' + sets['exercise'] + '|' + sets['date'].astype(str)
    return sets


def score(train, test):
    """MAE in reps for every predictor, trained on `train`, scored on `test`."""
    y = test['reps'].to_numpy()
    mae = lambda pred: float(np.mean(np.abs(y - np.asarray(pred))))
    carry = carry_reps(test['weight'], test['prior_set_weight'], test['prior_set_reps'])
    train_carry = carry_reps(train['weight'], train['prior_set_weight'], train['prior_set_reps'])

    legacy = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(train[LEGACY_FEATURES], train['reps'])
    model = fit_default(train)
    return {
        'Epley from rolling e1RM': mae(test['formula_pred_reps']),
        'Same reps as last set': mae(test['prior_set_reps']),
        'Last set carried to new weight': mae(carry),
        '... minus median fatigue drop': mae(carry - (train_carry - train['reps']).median()),
        'Legacy ridge (absolute weights)': mae(legacy.predict(test[LEGACY_FEATURES])),
        'Baseline + learned correction': mae(model.predict_many(
            test['weight'], test['set_number'], test['prior_set_reps'],
            test['prior_set_weight'], test['rolling_e1rm'])),
    }


def time_split(sets, frac=0.7):
    """Per lifter: earliest `frac` of their dates train, the rest test."""
    cutoffs = sets.groupby('lifter')['date'].transform(lambda d: d.quantile(frac))
    return sets[sets['date'] < cutoffs], sets[sets['date'] >= cutoffs]


def report(sets):
    columns = {}
    folds = [score(sets.iloc[tr], sets.iloc[te]) for tr, te in GroupKFold(5).split(sets, groups=sets['session'])]
    columns['Session CV'] = {k: np.mean([f[k] for f in folds]) for k in folds[0]}
    columns['Time split'] = score(*time_split(sets))
    if sets['lifter'].nunique() >= 2:
        held = [score(sets[sets.lifter != l], sets[sets.lifter == l]) for l in sets['lifter'].unique()]
        columns['New lifter'] = {k: np.mean([h[k] for h in held]) for k in held[0]}

    names = list(columns)
    print('| Predictor | ' + ' | '.join(f'{n} MAE' for n in names) + ' |')
    print('|---|' + '---|' * len(names))
    for row in columns[names[0]]:
        print(f'| {row} | ' + ' | '.join(f'{columns[n][row]:.2f}' for n in names) + ' |')


def main():
    all_sets = load_lifters()
    sets = in_planner_range(all_sets)  # score where the planner actually uses the model
    print(f"{len(all_sets)} usable sets; {len(sets) / len(all_sets):.1%} within the planner's "
          f"+/-{MAX_WEIGHT_CHANGE:.0%} weight range, which is what's trained and scored")
    print(f"{len(sets)} usable sets, {sets['session'].nunique()} sessions, "
          f"{sets['exercise'].nunique()} lifts, {sets['lifter'].nunique()} lifter(s)\n")
    report(sets)

    model = fit_default(sets)  # the shipped model learns from every set
    model.save()
    print('\nCorrection coefficients (reps per unit of each feature):')
    for name, c, s in zip(model.features, model.coef, model.scale):
        print(f'  {name:16s} {c / s:+.3f}')
    print(f'  intercept        {model.intercept:+.3f}')
    print('\nSaved rep_model.json')


if __name__ == '__main__':
    main()
