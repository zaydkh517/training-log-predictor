import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_dataset

FEATURES = ['weight', 'set_number', 'prior_set_reps', 'prior_set_weight', 'rolling_e1rm', 'weight_pct_e1rm']
TARGET = 'reps'


def evaluate_rep_model(usable_sets, seed=42, alpha=1.0):
    """Session-level split (never set-level) -- sets from the same workout are
    highly correlated fatigue states and must never leak across train/test."""
    sessions = usable_sets[['exercise', 'date']].drop_duplicates().sample(frac=1, random_state=seed)
    n_test = int(len(sessions) * 0.2)
    test_sessions = sessions.iloc[:n_test]
    train_sessions = sessions.iloc[n_test:]

    train = usable_sets.merge(train_sessions, on=['exercise', 'date'])
    test = usable_sets.merge(test_sessions, on=['exercise', 'date'])

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    formula_mae = mean_absolute_error(y_test, test['formula_pred_reps'])
    prior_mae = mean_absolute_error(y_test, test['prior_set_reps'])

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    ridge_mae = mean_absolute_error(y_test, model.predict(X_test))

    print(f"seed={seed} (train={len(train)}, test={len(test)})")
    print(f"  Formula baseline MAE:   {formula_mae:.2f}")
    print(f"  Prior-set baseline MAE: {prior_mae:.2f}")
    print(f"  Ridge MAE:              {ridge_mae:.2f}")

    return model


def main():
    dataset = build_dataset('data/strong_export.csv', 'data/hevy_export.csv')
    usable_sets = dataset['usable_sets']

    print(f"Usable set-level rows (all exercises pooled): {len(usable_sets)}\n")

    # validate across several splits before trusting a single result
    for seed in range(5):
        evaluate_rep_model(usable_sets, seed=seed)

    # train the final model on a fixed split and persist it for the Flask API
    final_model = evaluate_rep_model(usable_sets, seed=42)
    joblib.dump(final_model, 'rep_model.joblib')
    print("\nSaved rep_model.joblib")


if __name__ == '__main__':
    main()
