"""Set-level rep prediction: a strength-formula baseline plus a learned correction.

The baseline carries the prior set's same-day strength to the new weight with the
Epley formula. A ridge regression then learns only the gap between that baseline
and what lifters actually do, from unitless features, so a curl and a squat
share one model.

The correction can be personalized: fitted on one lifter's upload, shrunk toward
the shipped default, so a short history stays close to the default and a long
one earns its own fit.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURES = ['log_set', 'pct_e1rm', 'weight_change', 'prior_set_reps', 'prior_readiness']
DEFAULT_ALPHA = 10.0
PERSONAL_ALPHA = 50.0        # extra shrinkage toward the default for personal fits
MIN_PERSONAL_SETS = 30       # below this, a personal fit is mostly noise
# the planner only moves within +/-20% of the prior set, so the model is trained on that
# range too; bigger jumps are machine switches, where pounds aren't comparable, and a
# few of them dominate a squared-error fit
MAX_WEIGHT_CHANGE = 0.20

DEFAULT_MODEL_PATH = Path(__file__).with_name('rep_model.json')


def epley_reps(weight, rolling_e1rm):
    """Reps a fresh lifter could do at `weight`, from the Epley formula inverted."""
    return 30 * (rolling_e1rm / weight - 1)


def carry_reps(weight, prior_weight, prior_reps):
    """Baseline: the prior set's implied same-day e1RM, carried to this set's weight."""
    return (prior_reps + 30) * prior_weight / weight - 30


def feature_frame(weight, set_number, prior_reps, prior_weight, rolling_e1rm):
    """Model inputs. Works on scalars or aligned pandas Series."""
    return pd.DataFrame({
        'log_set': np.log(set_number),
        'pct_e1rm': weight / rolling_e1rm,
        'weight_change': weight / prior_weight - 1,
        'prior_set_reps': prior_reps,
        # how the prior set compared with what the rolling e1RM said it should be
        'prior_readiness': prior_reps - epley_reps(prior_weight, rolling_e1rm),
    }, index=[0] if np.isscalar(weight) else None)


def in_planner_range(usable_sets):
    """Sets whose weight is within MAX_WEIGHT_CHANGE of the prior set."""
    change = usable_sets['weight'] / usable_sets['prior_set_weight'] - 1
    return usable_sets[change.abs() <= MAX_WEIGHT_CHANGE]


def training_frame(usable_sets):
    """Features, baseline and residual target for every usable set in the planner's range."""
    s = in_planner_range(usable_sets)
    X = feature_frame(s['weight'], s['set_number'], s['prior_set_reps'], s['prior_set_weight'], s['rolling_e1rm'])
    carry = carry_reps(s['weight'], s['prior_set_weight'], s['prior_set_reps'])
    return X, carry, s['reps'] - carry


@dataclass
class RepModel:
    """A linear correction on standardized features, small enough to send as JSON."""
    mean: list
    scale: list
    coef: list
    intercept: float
    source: str = 'default'   # 'default' or 'personal'
    n_sets: int = 0
    features: tuple = tuple(FEATURES)

    def correction(self, X):
        z = (X[list(self.features)].to_numpy(dtype=float) - np.asarray(self.mean)) / np.asarray(self.scale)
        return z @ np.asarray(self.coef) + self.intercept

    def predict_many(self, weight, set_number, prior_reps, prior_weight, rolling_e1rm):
        """Unclipped predictions for evaluation."""
        X = feature_frame(weight, set_number, prior_reps, prior_weight, rolling_e1rm)
        return np.asarray(carry_reps(weight, prior_weight, prior_reps)) + self.correction(X)

    def predict(self, weight, set_number, prior_reps, prior_weight, rolling_e1rm):
        """Reps for one planned set, kept physically possible: never negative, and never
        more than a fresh lifter could do at that weight."""
        raw = float(self.predict_many(weight, set_number, prior_reps, prior_weight, rolling_e1rm)[0])
        return max(0.0, min(raw, epley_reps(weight, rolling_e1rm)))

    def to_dict(self):
        d = asdict(self)
        d['features'] = list(self.features)
        return d

    @classmethod
    def from_dict(cls, d):
        if list(d['features']) != FEATURES:
            raise ValueError('rep model was built for a different feature set')
        if not (len(d['mean']) == len(d['scale']) == len(d['coef']) == len(FEATURES)):
            raise ValueError('rep model has the wrong number of coefficients')
        if any(s <= 0 for s in d['scale']):
            raise ValueError('rep model scale values must be positive')
        return cls(**{**d, 'features': tuple(d['features'])})

    def save(self, path=DEFAULT_MODEL_PATH):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_default(usable_sets, alpha=DEFAULT_ALPHA):
    """Fit the shipped model on pooled training data."""
    X, _, resid = training_frame(usable_sets)
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=alpha).fit(scaler.transform(X), resid)
    return RepModel(
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
        coef=ridge.coef_.tolist(), intercept=float(ridge.intercept_),
        source='default', n_sets=len(X),
    )


def fit_personal(usable_sets, default, alpha=PERSONAL_ALPHA):
    """Personalize the default model on one lifter's sets.

    Fits only what the default gets wrong for this lifter, on the default's own
    feature scaling, and ridge-shrinks that adjustment toward zero -- so with
    little data the result stays close to the default. Returns the default
    unchanged when there isn't enough data to say anything.
    """
    X, _, resid = training_frame(usable_sets)
    if len(X) < MIN_PERSONAL_SETS:
        return default
    z = (X[FEATURES].to_numpy(dtype=float) - np.asarray(default.mean)) / np.asarray(default.scale)
    remaining = resid.to_numpy() - default.correction(X)
    delta = Ridge(alpha=alpha).fit(z, remaining)
    return RepModel(
        mean=list(default.mean), scale=list(default.scale),
        coef=(np.asarray(default.coef) + delta.coef_).tolist(),
        intercept=float(default.intercept + delta.intercept_),
        source='personal', n_sets=len(X),
    )
